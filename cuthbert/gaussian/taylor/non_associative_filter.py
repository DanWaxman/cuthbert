"""Implements the non-associative linearized Taylor Kalman filter."""

from typing import NamedTuple

from jax import ShapeDtypeStruct, tree
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular

from cuthbert.gaussian.taylor.types import (
    GetDynamicsLogDensity,
    GetObservationFunc,
    LogConditionalDensity,
    LogPotential,
)
from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbert.gaussian.utils import linearized_kalman_filter_state_dummy_elem
from cuthbert.utils import dummy_tree_like
from cuthbertlib.kalman import filtering
from cuthbertlib.linalg import block_marginal_sqrt_cov
from cuthbertlib.linearize import linearize_log_density, linearize_taylor
from cuthbertlib.types import Array, ArrayLike, ArrayTreeLike, KeyArray, LogDensity


class _TaylorPotential(NamedTuple):
    """Expansion point and log value of a Taylor observation potential."""

    anchor: Array
    log_value: Array


def process_observation(
    observation_output: tuple[LogConditionalDensity, Array, Array]
    | tuple[LogPotential, Array],
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array, Array, Array, _TaylorPotential | None]:
    r"""Prepare a Gaussian observation, retaining the scale of a log potential.

    Conditional log densities produce the usual observation parameters and no
    potential metadata. A log potential instead produces a pseudo-observation
    with $H=-I$, $y=0$, mean $d$ and covariance $R=LL^\top$. On active coordinates
    with nonsingular covariance, its approximation is

    $$
    \log G(x) \approx \log G(a)
      + (d-a)^\top R^{-1}(x-a) - \tfrac12(x-a)^\top R^{-1}(x-a).
    $$

    The expansion point $a$ and value $\log G(a)$ are retained so that evidence
    can be evaluated without subtracting large completed-square terms when
    curvature is weak. The covariance factor defines the effective curvature,
    including the regularisation performed by `linearize_taylor`.

    An all-NaN expansion point skips the observation, including its log value.
    With `ignore_nan_dims=True`, NaN covariance diagonals mark inactive
    coordinates; a constant potential still contributes its log value.

    Returns:
        Observation matrix, offset, covariance factor, observation, and optional
        Taylor potential metadata for evaluating the log evidence.
    """
    if len(observation_output) == 3:
        observation_cond_log_density, linearization_point, observation = (
            observation_output
        )
        H, d, chol_R = linearize_log_density(
            observation_cond_log_density,
            linearization_point,
            observation,
            rtol=rtol,
            ignore_nan_dims=ignore_nan_dims,
        )
        return H, d, chol_R, observation, None

    observation_log_potential, linearization_point = observation_output
    skip = jnp.isnan(linearization_point).all()
    anchor = jnp.where(skip, 0.0, linearization_point)

    def log_potential_with_value(x: ArrayLike) -> tuple[Array, Array]:
        value = observation_log_potential(x)
        return value, value

    d, chol_R, value = linearize_taylor(
        log_potential_with_value,
        anchor,
        has_aux=True,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )
    # With H = -I and y = 0, the pseudo-observation residual is x - d.
    H = -jnp.eye(d.shape[0])
    missing = skip | (ignore_nan_dims & jnp.isnan(jnp.diag(chol_R)))
    observation = jnp.where(missing, jnp.nan, 0.0)
    potential = _TaylorPotential(anchor, jnp.where(skip, 0.0, value))
    return H, d, chol_R, observation, potential


def _potential_log_evidence(
    mean: Array,
    chol_cov: Array,
    d: Array,
    chol_R: Array,
    observation: Array,
    potential: _TaylorPotential,
    fallback: Array,
) -> Array:
    r"""Integrate the anchored quadratic against a Gaussian without cancellation.

    Let $A=L_R^{-1}L_P$, $s=L_R^{-1}(\mu-a)$ and $z=L_R^{-1}(d-a)$.
    For $B=I+A^\top A$ and $u=A^\top(z-s)$, the log integral is

    $$
    \log G(a)+z^\top s-\tfrac12s^\top s
      -\tfrac12\log\det B+\tfrac12u^\top B^{-1}u.
    $$

    This never forms the large $z^\top z$ term. Masking uses the active block
    of the existing Taylor factor, so the evidence and state update use the
    same approximation. Singular or nonfinite active factors retain the
    uncorrected Gaussian evidence, matching the previous fallback.
    """
    missing = jnp.isnan(observation)
    # The active block of the Taylor factor is already lower triangular.
    active_factor = jnp.where(missing[:, None] | missing[None, :], 0.0, chol_R)
    valid = jnp.isfinite(active_factor).all() & jnp.all(
        missing | (jnp.diag(active_factor) != 0)
    )
    ignored = missing | ~valid
    factor = jnp.where(valid, active_factor, 0.0) + jnp.diag(
        ignored.astype(chol_R.dtype)
    )
    A = solve_triangular(factor, jnp.where(ignored[:, None], 0.0, chol_cov), lower=True)
    s = solve_triangular(
        factor, jnp.where(ignored, 0.0, mean - potential.anchor), lower=True
    )
    z = solve_triangular(
        factor, jnp.where(ignored, 0.0, d - potential.anchor), lower=True
    )
    chol_B = jnp.linalg.cholesky(jnp.eye(A.shape[1], dtype=A.dtype) + A.T @ A)
    whitened_u = solve_triangular(chol_B, A.T @ (z - s), lower=True)
    ell = (
        potential.log_value
        + z @ s
        - 0.5 * (s @ s)
        - jnp.log(jnp.diag(chol_B)).sum()
        + 0.5 * (whitened_u @ whitened_u)
    )
    return jnp.where(valid, ell, fallback)


def init_prepare(
    init_log_density: LogDensity,
    init_linearization_point: ArrayLike,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """Prepare the initial state for the linearized Taylor Kalman filter.

    Args:
        init_log_density: Initial log density log p(x_0). For factorial models,
            this is the sum of the log densities for all factors.
        init_linearization_point: Linearization point for the initial log density.
        rtol: The relative tolerance for the singular values of precision matrices
            when passed to `symmetric_inv_sqrt` during linearization.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            The default is determined based on the floating point precision of the dtype.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal of the
            precision matrices (found via linearization) as missing and ignore all rows
            and columns associated with them.
        key: JAX random key - not used.

    Returns:
        State for the linearized Taylor Kalman filter.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_normalizing_constant.
    """
    linearization_point = jnp.asarray(init_linearization_point)

    # Handle factorial axis if present
    linearization_shape = linearization_point.shape
    flat_linearization_point = linearization_point.reshape(-1)

    def flat_init_log_density(_, flat_x):
        return init_log_density(flat_x.reshape(linearization_shape))

    _, m0, chol_P0 = linearize_log_density(
        flat_init_log_density,
        flat_linearization_point,
        flat_linearization_point,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )
    m0 = m0.reshape(linearization_shape)

    if linearization_point.ndim > 1:
        chol_P0 = block_marginal_sqrt_cov(chol_P0, linearization_point.shape[-1])

    prior_state = linearized_kalman_filter_state_dummy_elem(
        mean=m0,
        chol_cov=chol_P0,
        log_normalizing_constant=jnp.array(0.0),
        model_inputs=None,
        mean_prev=dummy_tree_like(m0),
    )
    return prior_state


def filter_prepare(
    model_inputs: ArrayTreeLike,
    array_to_infer_shape: ArrayLike,
    key: KeyArray | None = None,
) -> LinearizedKalmanFilterState:
    """Prepare a state for a linearized Taylor Kalman filter step.

    Just passes through model inputs.

    Args:
        model_inputs: Model inputs.
        array_to_infer_shape: Array with shape matching state or mean, used to infer the
            state shape.
        key: JAX random key - not used.

    Returns:
        Prepared state for linearized Taylor Kalman filter.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    dummy_mean = dummy_tree_like(jnp.asarray(array_to_infer_shape))
    dummy_chol_cov = dummy_tree_like(
        ShapeDtypeStruct(dummy_mean.shape + dummy_mean.shape[-1:], dummy_mean.dtype)
    )

    return linearized_kalman_filter_state_dummy_elem(
        mean=dummy_mean,
        chol_cov=dummy_chol_cov,
        log_normalizing_constant=jnp.array(0.0),
        model_inputs=model_inputs,
        mean_prev=dummy_mean,
    )


def filter_combine(
    state_1: LinearizedKalmanFilterState,
    state_2: LinearizedKalmanFilterState,
    get_dynamics_log_density: GetDynamicsLogDensity,
    get_observation_func: GetObservationFunc,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> LinearizedKalmanFilterState:
    """Combine previous filter state with state prepared from latest model inputs.

    Applies linearized Taylor Kalman predict + filter update in covariance square
    root form.
    Not suitable for associative scan.

    Args:
        state_1: State from previous time step.
        state_2: State prepared (only access model_inputs attribute).
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
        get_observation_func: Function to get observation function (either conditional
            log density or log potential), linearization point and optional observation
            (not required for log potential functions).
        rtol: The relative tolerance for the singular values of precision matrices
            when passed to `symmetric_inv_sqrt` during linearization.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            The default is determined based on the floating point precision of the dtype.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal of the
            precision matrices (found via linearization) as missing and ignore all rows
            and columns associated with them.

    Returns:
        Predicted and updated linearized Taylor Kalman filter state.
            Contains mean, chol_cov (generalised Cholesky factor of covariance)
            and log_normalizing_constant.
    """
    log_dynamics_density, linearization_point_prev, linearization_point_curr = (
        get_dynamics_log_density(state_1, state_2.model_inputs)
    )

    F, c, chol_Q = linearize_log_density(
        log_dynamics_density,
        linearization_point_prev,
        linearization_point_curr,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )

    # If both linearization points are all NaN, skip the predict step
    skip_predict_flag = jnp.logical_and(
        jnp.all(jnp.isnan(linearization_point_curr)),
        jnp.all(jnp.isnan(linearization_point_prev)),
    )
    F = jnp.where(skip_predict_flag, jnp.eye(F.shape[0]), F)
    c = jnp.where(skip_predict_flag, jnp.zeros_like(c), c)
    chol_Q = jnp.where(skip_predict_flag, jnp.zeros_like(chol_Q), chol_Q)

    predict_mean, predict_chol_cov = filtering.predict(
        state_1.mean, state_1.chol_cov, F, c, chol_Q
    )

    predict_state = linearized_kalman_filter_state_dummy_elem(
        mean=predict_mean,
        chol_cov=predict_chol_cov,
        log_normalizing_constant=state_1.log_normalizing_constant,
        model_inputs=state_2.model_inputs,
        mean_prev=state_1.mean,
    )

    observation_output = get_observation_func(predict_state, state_2.model_inputs)

    H, d, chol_R, observation, potential = process_observation(
        observation_output,
        rtol=rtol,
        ignore_nan_dims=ignore_nan_dims,
    )

    (update_mean, update_chol_cov), log_normalizing_constant = filtering.update(
        predict_mean,
        predict_chol_cov,
        H,
        d,
        chol_R,
        observation,
    )
    if potential is not None:
        log_normalizing_constant = _potential_log_evidence(
            predict_mean,
            predict_chol_cov,
            d,
            chol_R,
            observation,
            potential,
            log_normalizing_constant,
        )

    return linearized_kalman_filter_state_dummy_elem(
        mean=update_mean,
        chol_cov=update_chol_cov,
        log_normalizing_constant=state_1.log_normalizing_constant
        + log_normalizing_constant,
        model_inputs=state_2.model_inputs,
        mean_prev=state_1.mean,
    )
