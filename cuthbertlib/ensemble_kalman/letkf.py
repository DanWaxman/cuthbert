"""Local Ensemble Transform Kalman Filter (LETKF) analysis.

See Hunt, Kostelich, and Szunyogh (2007),
https://doi.org/10.1016/j.physd.2006.11.008.
"""

import jax
import jax.numpy as jnp

from cuthbertlib.ensemble_kalman.filtering import (
    ObservationFn,
    _ensemble_log_likelihood,
)
from cuthbertlib.linalg import tria
from cuthbertlib.types import Array, ScalarArray


@jax.custom_jvp
def _symmetric_inverse_sqrt(matrix: Array) -> Array:
    """Return the symmetric inverse square root of a positive-definite matrix."""
    eigenvalues, basis = jnp.linalg.eigh(matrix)
    return (basis / jnp.sqrt(eigenvalues)) @ basis.T


@_symmetric_inverse_sqrt.defjvp
def _symmetric_inverse_sqrt_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    """Differentiate the matrix function without eigenvalue-gap divisions."""
    (matrix,), (direction,) = primals, tangents
    eigenvalues, basis = jnp.linalg.eigh(matrix)
    roots = jnp.sqrt(eigenvalues)
    value = (basis / roots) @ basis.T
    # The divided difference of lambda ** (-1/2) has this continuous form,
    # including equal eigenvalues. eigh symmetrizes its input, so do so here too.
    direction = (direction + direction.T) / 2
    denominator = roots[:, None] * roots[None, :] * (roots[:, None] + roots[None, :])
    derivative = basis @ (-(basis.T @ direction @ basis) / denominator) @ basis.T
    return value, derivative


def update(
    predicted_ensemble: Array,
    observation_fn: ObservationFn,
    chol_R: Array,
    y: Array,
    observation_indices: Array,
    localization_weights: Array,
) -> tuple[Array, ScalarArray]:
    r"""Apply local deterministic transforms and score the global forecast.

    The forecast is scored once as a Gaussian with observation-ensemble mean
    and covariance ``R + Y @ Y.T``, where ``Y`` is the observation anomaly
    matrix divided by ``sqrt(N - 1)``. This is a normalized global predictive
    density approximation, independent of the current localization weights;
    it is not an unbiased estimate of the original model's likelihood.

    Each state coordinate then uses a local ensemble-space transform, with
    observation precision weighted by ``localization_weights``. NaNs in ``y``
    indicate missing observations. Padding, zero weights, and missing entries
    are excluded. A coordinate with no active observations is unchanged; if
    every observation is missing, the log-likelihood contribution is zero.

    Args:
        predicted_ensemble: Forecast members, shape (N, x_dim), with N >= 2.
        observation_fn: Function mapping a state vector to shape (y_dim,).
        chol_R: Scalar observation-noise factor or diagonal factor of shape
            (y_dim,). Factors for observed dimensions must be finite and
            nonzero. Two-dimensional factors are not supported.
        y: Nonempty observation vector, shape (y_dim,).
        observation_indices: Integer indices in the original observation order,
            shape (x_dim, max_local_observations). Use -1 for padding; other
            entries must be in range and unique within each row.
        localization_weights: Finite, nonnegative precision weights with the
            same shape as ``observation_indices``. Weights at padded indices
            are ignored. Neighborhood shapes must remain fixed under JIT;
            indices and weights may vary with model inputs.

    Returns:
        Updated ensemble of shape (N, x_dim) and the global predictive log density.

    Raises:
        ValueError: If the ensemble has fewer than two members, indices are not
            integers, or input shapes are incompatible.
    """
    predicted_ensemble = jnp.asarray(predicted_ensemble)
    chol_R, y = jnp.asarray(chol_R), jnp.asarray(y)
    observation_indices = jnp.asarray(observation_indices)
    localization_weights = jnp.asarray(localization_weights)
    if predicted_ensemble.ndim != 2 or predicted_ensemble.shape[0] < 2:
        raise ValueError(
            "predicted_ensemble must have shape (N, x_dim) with N at least 2."
        )
    if y.ndim != 1 or y.shape[0] == 0:
        raise ValueError("y must be a nonempty vector of shape (y_dim,).")
    if chol_R.ndim != 0 and chol_R.shape != y.shape:
        raise ValueError("chol_R must be a scalar or a 1D factor of shape (y_dim,).")
    if (
        observation_indices.ndim != 2
        or observation_indices.shape[0] != predicted_ensemble.shape[1]
        or localization_weights.shape != observation_indices.shape
    ):
        raise ValueError(
            "observation_indices and localization_weights must both have shape "
            "(x_dim, max_local_observations)."
        )
    if not jnp.issubdtype(observation_indices.dtype, jnp.integer):
        raise ValueError("observation_indices must contain integers.")

    n_particles = predicted_ensemble.shape[0]
    scale = jnp.sqrt(jnp.asarray(n_particles - 1, dtype=predicted_ensemble.dtype))
    y_pred = jax.vmap(observation_fn)(predicted_ensemble)
    if y_pred.shape != (n_particles, y.shape[0]):
        raise ValueError(
            "observation_fn must return a vector with the same shape as y."
        )

    observed = ~jnp.isnan(y)
    # Mask before arithmetic: multiplying missing predictions by zero is not safe.
    y_pred = jnp.where(observed[None, :], y_pred, 0)
    y_mean = jnp.mean(y_pred, axis=0)
    anomalies = (y_pred - y_mean).T / scale
    innovation = jnp.where(observed, y, 0) - y_mean
    noise = jnp.broadcast_to(chol_R, y.shape)

    # Missing dimensions cancel their own Gaussian normalization, as in EnKF.
    score_noise = jnp.where(observed, noise, 1 / jnp.sqrt(2 * jnp.pi))
    whitened_anomalies = anomalies / score_noise[:, None]
    whitened_innovation = innovation / score_noise
    identity = jnp.eye(n_particles, dtype=whitened_anomalies.dtype)
    chol_C = tria(jnp.concatenate([whitened_anomalies.T, identity], axis=1))
    ll = _ensemble_log_likelihood(
        whitened_anomalies, whitened_innovation, score_noise, chol_C
    )
    ll = jnp.where(jnp.any(observed), ll, 0)

    def analyze_coordinate(inputs: tuple[Array, Array, Array]) -> Array:
        forecast, indices, weights = inputs
        safe_indices = jnp.maximum(indices, 0)
        active = (indices >= 0) & observed[safe_indices] & (weights > 0)
        local_anomalies = jnp.where(active[:, None], anomalies[safe_indices], 0)
        local_innovation = jnp.where(active, innovation[safe_indices], 0)
        local_noise = jnp.where(active, noise[safe_indices], 1)
        precision = jnp.where(active, weights, 0) / jnp.square(local_noise)
        C = identity + local_anomalies.T @ (precision[:, None] * local_anomalies)
        coefficients = jnp.linalg.solve(
            C, local_anomalies.T @ (precision * local_innovation)
        )
        transform = _symmetric_inverse_sqrt(C)
        deviations = forecast - jnp.mean(forecast)
        # Express as increments so an identity transform preserves members exactly.
        return (
            forecast
            + deviations @ coefficients / scale
            + deviations @ (transform - identity)
        )

    updated = jax.lax.map(
        analyze_coordinate,
        (predicted_ensemble.T, observation_indices, localization_weights),
    ).T
    return updated, ll
