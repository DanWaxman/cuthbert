"""Global and local Ensemble Transform Kalman Filter analysis.

See Hunt, Kostelich, and Szunyogh (2007),
https://doi.org/10.1016/j.physd.2006.11.008.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from cuthbertlib.ensemble_kalman.filtering import (
    ObservationFn,
    _ensemble_log_likelihood,
    _whiten,
)
from cuthbertlib.linalg import collect_nans_chol, tria
from cuthbertlib.types import Array, ScalarArray


@jax.custom_jvp
def _symmetric_inverse_sqrt_from_factor(factor: Array) -> Array:
    """Return the symmetric inverse square root of factor @ factor.T."""
    basis, roots, _ = jnp.linalg.svd(factor, full_matrices=False)
    return (basis / roots) @ basis.T


@_symmetric_inverse_sqrt_from_factor.defjvp
def _symmetric_inverse_sqrt_from_factor_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    """Differentiate through the implied Gramian without singular-value gaps."""
    (factor,), (direction,) = primals, tangents
    basis, roots, _ = jnp.linalg.svd(factor, full_matrices=False)
    value = (basis / roots) @ basis.T
    gram_direction = direction @ factor.T
    gram_direction = gram_direction + gram_direction.T
    # The divided difference of lambda**(-1/2) extends to equal eigenvalues.
    denominator = roots[:, None] * roots[None, :] * (roots[:, None] + roots[None, :])
    tangent = basis @ (-(basis.T @ gram_direction @ basis) / denominator) @ basis.T
    return value, tangent


def _analysis(
    whitened_anomalies: Array, whitened_innovation: Array
) -> tuple[Array, Array, Array]:
    """Return mean weights, the symmetric transform, and the precision factor."""
    n_members = whitened_anomalies.shape[1]
    dtype = jnp.result_type(whitened_anomalies, whitened_innovation)
    # Hunt et al. (2007), Eq. (19), in normalized, whitened coordinates.
    # The first N rows encode [Z.T, I]. The last row encodes the target,
    # with an extra unit column to keep the augmented matrix full row rank.
    # QR puts the target projected onto the design basis in the last row of L.
    augmented = jnp.concatenate(
        (
            jnp.concatenate(
                (whitened_anomalies.T, whitened_innovation[None, :]), axis=0
            ),
            jnp.eye(n_members + 1, dtype=dtype),
        ),
        axis=1,
    )
    lower = tria(augmented)
    # The leading block factors I + Z.T @ Z, the scaled precision in Eq. (21).
    factor = lower[:n_members, :n_members]
    # Eq. (20). QR gives the projected target for the triangular solve.
    coefficients = solve_triangular(factor.T, lower[n_members, :n_members])
    # Eq. (24). The SVD recovers the symmetric transform from that factor.
    return coefficients, _symmetric_inverse_sqrt_from_factor(factor), factor


def update(
    predicted_ensemble: Array,
    observation_fn: ObservationFn,
    chol_R: Array,
    y: Array,
    observation_indices: Array | None = None,
    localization_weights: Array | None = None,
    batch_size: int | None = None,
) -> tuple[Array, ScalarArray]:
    r"""Apply a deterministic ensemble transform and score the global forecast.

    The forecast is scored once as a Gaussian with observation-ensemble mean
    and covariance ``R + Y @ Y.T``, where ``Y`` is the observation anomaly
    matrix divided by ``sqrt(N - 1)``. This is a normalized global predictive
    density approximation, independent of the current localization weights;
    it is not an unbiased estimate of the original model's likelihood.

    With no localization, one transform is shared by all state coordinates.
    When both localization arrays are supplied, each state coordinate uses
    its own ensemble-space transform, with
    observation precision weighted by ``localization_weights``. NaNs in ``y``
    indicate missing observations. Padding, zero weights, and missing entries
    are excluded. A coordinate with no active observations is unchanged; if
    every observation is missing, the log-likelihood contribution is zero.

    Args:
        predicted_ensemble: Forecast members, shape (N, x_dim), with N >= 2.
        observation_fn: Function mapping a state vector to shape (y_dim,).
        chol_R: Scalar observation-noise factor or diagonal factor of shape
            (y_dim,). Global ETKF also accepts a dense lower triangular factor
            of shape (y_dim, y_dim). Factors for observed dimensions must be
            finite and nonzero on the diagonal.
        y: Nonempty observation vector, shape (y_dim,).
        observation_indices: Integer indices in the original observation order,
            shape (x_dim, max_local_observations). Use -1 for padding; other
            entries must be in range and unique within each row. Supply with
            localization_weights to enable localization.
        localization_weights: Finite, nonnegative precision weights with the
            same shape as ``observation_indices``. Weights at padded indices
            are ignored. Neighborhood shapes must remain fixed under JIT;
            indices and weights may vary with model inputs.
        batch_size: Number of state coordinates to analyze concurrently in
            ``jax.lax.map``. ``None`` processes one coordinate at a time.
            Larger batches trade memory for parallelism. Applies only with
            localization and must be static under JIT.

    Returns:
        Updated ensemble of shape (N, x_dim) and the global predictive log density.

    Raises:
        ValueError: If the ensemble has fewer than two members, indices are not
            integers, only one localization array is provided, or input shapes
            are incompatible.
    """
    predicted_ensemble = jnp.asarray(predicted_ensemble)
    chol_R, y = jnp.asarray(chol_R), jnp.asarray(y)
    if predicted_ensemble.ndim != 2 or predicted_ensemble.shape[0] < 2:
        raise ValueError(
            "predicted_ensemble must have shape (N, x_dim) with N at least 2."
        )
    if y.ndim != 1 or y.shape[0] == 0:
        raise ValueError("y must be a nonempty vector of shape (y_dim,).")
    if (observation_indices is None) != (localization_weights is None):
        raise ValueError("Supply both observation_indices and localization_weights.")
    if chol_R.ndim == 2:
        if observation_indices is not None:
            raise ValueError(
                "Localized ETKF requires a scalar or 1D diagonal chol_R factor."
            )
        if chol_R.shape != (y.size, y.size):
            raise ValueError("Dense chol_R must have shape (y_dim, y_dim).")
    elif chol_R.ndim != 0 and chol_R.shape != y.shape:
        raise ValueError(
            "chol_R must be scalar or have shape (y_dim,) or (y_dim, y_dim)."
        )
    if observation_indices is not None and localization_weights is not None:
        observation_indices = jnp.asarray(observation_indices)
        localization_weights = jnp.asarray(localization_weights)
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
    # Eq. (17). Apply the observation operator to each forecast member.
    y_pred = jax.vmap(observation_fn)(predicted_ensemble)
    if y_pred.shape != (n_particles, y.shape[0]):
        raise ValueError(
            "observation_fn must return a vector with the same shape as y."
        )

    observed = ~jnp.isnan(y)
    # Mask before arithmetic: multiplying missing predictions by zero is not safe.
    y_pred = jnp.where(observed[None, :], y_pred, 0)
    y_mean = jnp.mean(y_pred, axis=0)
    # Normalize the observation perturbations used in Eq. (18).
    anomalies = (y_pred - y_mean).T / scale
    innovation = jnp.where(observed, y, 0) - y_mean
    # Missing dimensions cancel their own Gaussian normalization, as in EnKF.
    if chol_R.ndim == 2:
        chol_R = chol_R.astype(jnp.result_type(chol_R, 1.0))
        score_noise, innovation, anomalies = jax.lax.cond(
            jnp.all(observed),
            lambda args: args,
            lambda args: collect_nans_chol(~observed, *args)[1:],
            (chol_R, innovation, anomalies),
        )
    else:
        score_noise = jnp.where(observed, chol_R, 1 / jnp.sqrt(2 * jnp.pi))
    whitened_anomalies = _whiten(score_noise, anomalies)
    whitened_innovation = _whiten(score_noise, innovation)
    identity = jnp.eye(n_particles, dtype=whitened_anomalies.dtype)

    def apply_transform(
        forecast: Array, coefficients: Array, transform: Array
    ) -> Array:
        deviations = forecast - jnp.mean(forecast, axis=-1, keepdims=True)
        # Eqs. (22) and (25). Convert the mean weights to the paper's scaling.
        # Express as increments so an identity transform preserves members exactly.
        return (
            forecast
            + (deviations @ coefficients)[..., None] / scale
            + deviations @ (transform - identity)
        )

    if observation_indices is None or localization_weights is None:
        coefficients, transform, chol_C = _analysis(
            whitened_anomalies, whitened_innovation
        )
        updated = apply_transform(predicted_ensemble.T, coefficients, transform).T
    else:
        noise = jnp.broadcast_to(chol_R, y.shape)
        chol_C = tria(jnp.concatenate([whitened_anomalies.T, identity], axis=1))

        def analyze_coordinate(inputs: tuple[Array, Array, Array]) -> Array:
            forecast, indices, weights = inputs
            safe_indices = jnp.maximum(indices, 0)
            active = (indices >= 0) & observed[safe_indices] & (weights > 0)
            local_anomalies = jnp.where(active[:, None], anomalies[safe_indices], 0)
            local_innovation = jnp.where(active, innovation[safe_indices], 0)
            local_noise = jnp.where(active, noise[safe_indices], 1)
            # Section 2.3.4 scales R^{-1} by the local precision weights.
            # Mask before sqrt to keep the derivative finite at excluded weights.
            whiten = jnp.where(active, jnp.sqrt(jnp.where(active, weights, 1)), 0)
            whiten = whiten / local_noise
            coefficients, transform, _ = _analysis(
                whiten[:, None] * local_anomalies, whiten * local_innovation
            )
            return apply_transform(forecast, coefficients, transform)

        updated = jax.lax.map(
            analyze_coordinate,
            (predicted_ensemble.T, observation_indices, localization_weights),
            batch_size=batch_size,
        ).T

    ll = _ensemble_log_likelihood(
        whitened_anomalies, whitened_innovation, score_noise, chol_C
    )
    return updated, jnp.where(jnp.any(observed), ll, 0)
