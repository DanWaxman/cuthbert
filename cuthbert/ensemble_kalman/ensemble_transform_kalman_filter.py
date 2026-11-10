"""Implements the high-level Ensemble Transform Kalman Filter (ETKF)."""

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
    GetETKFLocalization,
    InitSample,
)
from cuthbert.inference import Filter
from cuthbertlib import ensemble_kalman as enkf_lib


def build_filter(
    init_sample: InitSample,
    get_dynamics: GetEnKFDynamics,
    get_observations: GetEnKFObservations,
    n_particles: int,
    get_localization: GetETKFLocalization | None = None,
    inflation: float = 0.0,
    store_predicted_ensemble: bool = False,
    batch_size: int | None = None,
) -> Filter:
    """Build an ETKF with optional localization and a global Gaussian predictive likelihood.

    The analysis deterministically transforms the forecast ensemble. Providing
    localization applies a separate transform at each state coordinate. The log normalizing constant
    accumulates a global Gaussian forecast density, rather than a sum of local
    scores. This is an approximate model likelihood, not an unbiased estimator.

    Args:
        init_sample: Function of a JAX key sampling one initial state vector.
        get_dynamics: Function of model inputs returning a simulator
            ``(state, key) -> next_state``.
        get_observations: Function of model inputs returning the observation
            function, noise factor ``chol_R``, and observation vector. Global
            ETKF accepts a scalar, diagonal vector, or lower triangular matrix.
            Localized ETKF requires a scalar or diagonal vector.
            NaNs in the observation vector indicate missing dimensions.
        n_particles: Ensemble size, at least two.
        get_localization: Optional function of model inputs returning padded observation
            indices and precision weights, both shaped
            (x_dim, max_local_observations). Indices refer to the original
            observation order; -1 marks padding. Other indices must be in range
            and unique within each row. Weights must be finite and nonnegative;
            zero weights exclude observations. Shapes must remain fixed under JIT.
            ``None`` selects global ETKF.
        inflation: Forecast deviations are multiplied by ``1 + inflation``,
            following the EnKF convention.
        store_predicted_ensemble: Whether to retain the incoming forecast ensemble
            in each state, following the EnKF convention.
        batch_size: Number of state coordinates to analyze concurrently in
            ``jax.lax.map``. ``None`` processes one coordinate at a time.
            Larger batches trade memory for parallelism. Applies only when localization is enabled.

    Returns:
        Non-associative filter using ``EnKFState`` and the usual EnKF key handling.

    Raises:
        ValueError: If ``n_particles`` is less than two.
    """
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2 for ETKF.")

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
    get_localization: GetETKFLocalization | None = None,
    inflation: float = 0.0,
    store_predicted_ensemble: bool = False,
    batch_size: int | None = None,
) -> EnKFState:
    """Propagate the previous ensemble and apply a global or local transform analysis.

    Args:
        state_1: Previous filter state.
        state_2: Prepared state containing the current model inputs.
        get_dynamics: Function getting the state simulator from model inputs.
        get_observations: Function getting the observation function, noise factor,
            and observation vector from model inputs.
        get_localization: Optional function getting local indices and precision weights.
        inflation: Multiplicative forecast anomaly inflation, as in EnKF.
        store_predicted_ensemble: Whether to retain the incoming forecast ensemble.
        batch_size: Number of state coordinates to analyze concurrently, or
            ``None`` to process one at a time. Applies only when localization is enabled.

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
    observation_indices, localization_weights = (
        (None, None)
        if get_localization is None
        else get_localization(state_2.model_inputs)
    )
    updated, ll = enkf_lib.etkf_update(
        predicted,
        observation_fn,
        chol_R,
        y,
        observation_indices,
        localization_weights,
        batch_size=batch_size,
    )
    return EnKFState(
        key=key_next,
        ensemble=updated,
        model_inputs=state_2.model_inputs,
        log_normalizing_constant=state_1.log_normalizing_constant + ll,
        predicted_ensemble=predicted if store_predicted_ensemble else None,
    )
