"""Local Ensemble Transform Kalman Filter (LETKF) analysis.

See Hunt, Kostelich, and Szunyogh (2007),
https://doi.org/10.1016/j.physd.2006.11.008.
Optional weight interpolation follows Yang et al. (2009),
https://doi.org/10.1002/qj.353.
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
    batch_size: int | None = None,
    weight_interpolation: tuple[Array, Array] | None = None,
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
    are excluded. Without interpolation, a coordinate with no active
    observations is unchanged. With interpolation, its analysis can inherit
    information from neighboring coarse analysis points. If every observation
    is missing, the ensemble is unchanged and the log-likelihood is zero.

    Args:
        predicted_ensemble: Forecast members, shape (N, x_dim), with N >= 2.
        observation_fn: Function mapping a state vector to shape (y_dim,).
        chol_R: Scalar observation-noise factor or diagonal factor of shape
            (y_dim,). Factors for observed dimensions must be finite and
            nonzero. Two-dimensional factors are not supported.
        y: Nonempty observation vector, shape (y_dim,).
        observation_indices: Integer indices in the original observation order,
            shape (n_analysis, max_local_observations). Without interpolation,
            n_analysis must equal x_dim. With interpolation, rows describe the
            coarse analysis points. Use -1 for padding; other entries must be
            in range and unique within each row.
        localization_weights: Finite, nonnegative precision weights with the
            same shape as ``observation_indices``. Weights at padded indices
            are ignored. Neighborhood shapes must remain fixed under JIT;
            indices and weights may vary with model inputs.
        batch_size: Number of state coordinates to analyze concurrently in
            ``jax.lax.map``. ``None`` processes one coordinate at a time.
            Larger batches trade memory for parallelism. Must be static under JIT.
            With interpolation, applies to both coarse solves and fine updates.
        weight_interpolation: Optional pair of coarse-analysis indices and
            interpolation coefficients, both shaped (x_dim, stencil_size).
            Indices refer to rows of ``observation_indices``; -1 marks padding.
            Nonpadding indices must be in range. Coefficients must be finite,
            nonnegative, and sum to one over nonpadding entries in each row.
            Padding coefficients are ignored. Both the mean-analysis weights
            and symmetric transforms are interpolated and applied to each fine
            coordinate's own forecast anomalies, following Yang et al. (2009).
            ``None`` disables interpolation. Shapes must remain fixed under JIT;
            index and coefficient value constraints are caller requirements.

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
        or localization_weights.shape != observation_indices.shape
    ):
        raise ValueError(
            "observation_indices and localization_weights must both have shape "
            "(n_analysis, max_local_observations)."
        )
    if not jnp.issubdtype(observation_indices.dtype, jnp.integer):
        raise ValueError("observation_indices must contain integers.")
    if weight_interpolation is None:
        if observation_indices.shape[0] != predicted_ensemble.shape[1]:
            raise ValueError("Without interpolation, n_analysis must equal x_dim.")
    else:
        interpolation_indices, interpolation_weights = (
            jnp.asarray(value) for value in weight_interpolation
        )
        if (
            observation_indices.shape[0] == 0
            or interpolation_indices.ndim != 2
            or interpolation_indices.shape[0] != predicted_ensemble.shape[1]
            or interpolation_weights.shape != interpolation_indices.shape
        ):
            raise ValueError(
                "weight_interpolation requires at least one analysis point and "
                "indices and coefficients of shape (x_dim, stencil_size)."
            )
        if not jnp.issubdtype(interpolation_indices.dtype, jnp.integer):
            raise ValueError("Interpolation indices must contain integers.")

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

    def compute_weights(inputs: tuple[Array, Array]) -> tuple[Array, Array]:
        indices, weights = inputs
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
        # Interpolating W - I is equivalent to interpolating W when stencil
        # coefficients sum to one, and preserves a no-observation identity exactly.
        return coefficients / scale, _symmetric_inverse_sqrt(C) - identity

    def analyze_coordinate(inputs: tuple[Array, Array, Array]) -> Array:
        forecast, indices, weights = inputs
        mean_weights, transform_increment = compute_weights((indices, weights))
        deviations = forecast - jnp.mean(forecast)
        # Express as increments so an identity transform preserves members exactly.
        return forecast + deviations @ mean_weights + deviations @ transform_increment

    if weight_interpolation is None:
        updated = jax.lax.map(
            analyze_coordinate,
            (predicted_ensemble.T, observation_indices, localization_weights),
            batch_size=batch_size,
        ).T
    else:
        coarse_mean_weights, coarse_transform_increments = jax.lax.map(
            compute_weights,
            (observation_indices, localization_weights),
            batch_size=batch_size,
        )

        def analyze_interpolated(inputs: tuple[Array, Array, Array]) -> Array:
            forecast, indices, weights = inputs
            safe_indices = jnp.maximum(indices, 0)
            weights = jnp.where(indices >= 0, weights, 0)
            mean_weights = weights @ coarse_mean_weights[safe_indices]
            transform_increment = jnp.einsum(
                "s,sij->ij", weights, coarse_transform_increments[safe_indices]
            )
            deviations = forecast - jnp.mean(forecast)
            return (
                forecast + deviations @ mean_weights + deviations @ transform_increment
            )

        updated = jax.lax.map(
            analyze_interpolated,
            (predicted_ensemble.T, interpolation_indices, interpolation_weights),
            batch_size=batch_size,
        ).T
    return updated, ll
