"""Implements the high-level Local Ensemble Transform Kalman Filter (LETKF)."""

from functools import partial

from jax import random

from cuthbert.ensemble_kalman.ensemble_kalman_filter import (
    EnKFState,
    filter_prepare,
    init_prepare,
)
from cuthbert.ensemble_kalman.types import (
    GetEnKFDynamics,
    GetEnKFObservations,
    GetLETKFLocalization,
    GetLETKFWeightInterpolation,
    InitSample,
)
from cuthbert.inference import Filter
from cuthbertlib import ensemble_kalman as enkf_lib


def build_filter(
    init_sample: InitSample,
    get_dynamics: GetEnKFDynamics,
    get_observations: GetEnKFObservations,
    n_particles: int,
    get_localization: GetLETKFLocalization,
    inflation: float = 0.0,
    store_predicted_ensemble: bool = False,
    batch_size: int | None = None,
    get_weight_interpolation: GetLETKFWeightInterpolation | None = None,
) -> Filter:
    """Build a LETKF with a global Gaussian predictive likelihood.

    The analysis deterministically transforms the forecast ensemble at each
    state coordinate using its local observations. The log normalizing constant
    accumulates a global Gaussian forecast density, rather than a sum of local
    scores. This is an approximate model likelihood, not an unbiased estimator.

    Args:
        init_sample: Function of a JAX key sampling one initial state vector.
        get_dynamics: Function of model inputs returning a simulator
            ``(state, key) -> next_state``.
        get_observations: Function of model inputs returning the observation
            function, scalar or 1D diagonal ``chol_R``, and observation vector.
            NaNs in the observation vector indicate missing dimensions.
        n_particles: Ensemble size, at least two.
        get_localization: Function of model inputs returning padded observation
            indices and precision weights, both shaped
            (n_analysis, max_local_observations). Without weight interpolation,
            n_analysis equals x_dim; otherwise rows describe coarse analysis
            points. Indices refer to the original
            observation order; -1 marks padding. Other indices must be in range
            and unique within each row. Weights must be finite and nonnegative;
            zero weights exclude observations. Shapes must remain fixed under JIT.
        inflation: Forecast deviations are multiplied by ``1 + inflation``,
            following the EnKF convention.
        store_predicted_ensemble: Whether to retain the incoming forecast ensemble
            in each state, following the EnKF convention.
        batch_size: Number of state coordinates to analyze concurrently in
            ``jax.lax.map``. ``None`` processes one coordinate at a time.
            Larger batches trade memory for parallelism.
        get_weight_interpolation: Optional function of model inputs returning
            coarse-analysis indices and interpolation coefficients, both shaped
            (x_dim, stencil_size). Indices refer to the rows of
            ``get_localization``; -1 marks padding. Other indices must be in
            range. Coefficients must be finite, nonnegative, and sum to one over
            nonpadding entries in each row. Both mean-analysis weights and
            symmetric transforms are interpolated as in Yang et al. (2009).
            ``None`` keeps the standard per-coordinate analysis. Shapes must
            stay fixed under JIT; callers supply the interpolation geometry.

    Returns:
        Non-associative filter using ``EnKFState`` and the usual EnKF key handling.

    Raises:
        ValueError: If ``n_particles`` is less than two.
    """
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2 for LETKF.")

    return Filter(
        init_prepare=partial(
            init_prepare,
            init_sample=init_sample,
            n_particles=n_particles,
            store_predicted_ensemble=store_predicted_ensemble,
        ),
        filter_prepare=partial(
            filter_prepare,
            init_sample=init_sample,
            n_particles=n_particles,
            store_predicted_ensemble=store_predicted_ensemble,
        ),
        filter_combine=partial(
            filter_combine,
            get_dynamics=get_dynamics,
            get_observations=get_observations,
            get_localization=get_localization,
            get_weight_interpolation=get_weight_interpolation,
            batch_size=batch_size,
            inflation=inflation,
            store_predicted_ensemble=store_predicted_ensemble,
        ),
        associative=False,
    )


def filter_combine(
    state_1: EnKFState,
    state_2: EnKFState,
    get_dynamics: GetEnKFDynamics,
    get_observations: GetEnKFObservations,
    get_localization: GetLETKFLocalization,
    inflation: float = 0.0,
    store_predicted_ensemble: bool = False,
    batch_size: int | None = None,
    get_weight_interpolation: GetLETKFWeightInterpolation | None = None,
) -> EnKFState:
    """Propagate the previous ensemble and apply a local transform analysis.

    Args:
        state_1: Previous filter state.
        state_2: Prepared state containing the current model inputs.
        get_dynamics: Function getting the state simulator from model inputs.
        get_observations: Function getting the observation function, noise factor,
            and observation vector from model inputs.
        get_localization: Function getting local indices and precision weights.
        inflation: Multiplicative forecast anomaly inflation, as in EnKF.
        store_predicted_ensemble: Whether to retain the incoming forecast ensemble.
        batch_size: Number of state coordinates to analyze concurrently, or
            ``None`` to process one at a time.
        get_weight_interpolation: Optional callback returning coarse-analysis
            indices and interpolation coefficients for each state coordinate.

    Returns:
        Updated ensemble state with cumulative global predictive log likelihood.
    """
    key_pred, _, key_next = random.split(state_1.key, 3)
    predicted = enkf_lib.predict(
        key_pred,
        state_1.ensemble,
        get_dynamics(state_2.model_inputs),
        inflation,
    )
    observation_fn, chol_R, y = get_observations(state_2.model_inputs)
    observation_indices, localization_weights = get_localization(state_2.model_inputs)
    weight_interpolation = (
        None
        if get_weight_interpolation is None
        else get_weight_interpolation(state_2.model_inputs)
    )
    updated, ll = enkf_lib.letkf_update(
        predicted,
        observation_fn,
        chol_R,
        y,
        observation_indices,
        localization_weights,
        batch_size=batch_size,
        weight_interpolation=weight_interpolation,
    )
    return EnKFState(
        key=key_next,
        ensemble=updated,
        model_inputs=state_2.model_inputs,
        log_normalizing_constant=state_1.log_normalizing_constant + ll,
        predicted_ensemble=predicted if store_predicted_ensemble else None,
    )
