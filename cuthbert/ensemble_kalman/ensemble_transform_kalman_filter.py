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

    The ETKF ([Bishop, Etherton, and Majumdar (2001)](https://doi.org/10.1175/1520-0493%282001%29129%3C0420%3AASWTET%3E2.0.CO%3B2)) is a deterministic square-root EnKF variant. It works natively in the
    ensemble space, and is thus very efficient for small ensembles. Its localized variant,
    the LETKF ([Hunt, Kostelich, and Szunyogh (2007)](https://doi.org/10.1016/j.physd.2006.11.008)), uses `get_localization` to compute a separate,
    localized analysis for each state coordinate.

    The likelihood used to compute the `log_normalizing_constant` is the global Gaussian
    predictive log density of the full forecast observation ensemble, evaluated before each update step.

    Args:
        init_sample: Function of a JAX random key only, generates a single sample from
            the initial distribution.
        get_dynamics: Function to get dynamics function (x_t, key) -> x_{t+1} ~ p(x_{t+1} | x_t) from model inputs.
        get_observations: Function to get observation function, chol_R, and y from model inputs. Global
            ETKF accepts a scalar, diagonal vector, or lower triangular matrix.
            Localized ETKF requires a scalar or diagonal vector.
        n_particles: Number of particles.
        get_localization: Optional function of model inputs returning padded observation
            indices and precision weights, both shaped
            (x_dim, max_local_observations). Indices refer to the original
            observation order; -1 marks padding. Other indices must be in range
            and unique within each row. Weights must be finite and nonnegative;
            zero weights exclude observations. Shapes must remain fixed under JIT.
            ``None`` selects global ETKF.
        inflation: Multiplicative inflation factor for ensemble deviations, applied in the predict step.
        store_predicted_ensemble: Whether to store the incoming forecast ensemble
            in each filter state, as required by the EnRTS smoother.
        batch_size: Number of state coordinates to analyze concurrently in
            ``jax.lax.map``. ``None`` processes one coordinate at a time.
            Larger batches trade memory for parallelism. Applies only when localization is enabled.

    Returns:
        Non-associative filter using ``EnKFState`` and the usual EnKF key handling.

    Raises:
        ValueError: If ``n_particles`` is less than 2.
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
        state_1: EnKF state from the previous time step.
        state_2: EnKF state prepared for the current step.
        get_dynamics: Function to get dynamics function (x_t, key) -> x_{t+1} ~ p(x_{t+1} | x_t) from model inputs.
        get_observations: Function to get observation function, chol_R, and y from model inputs.
        get_localization: Optional function getting local indices and precision weights.
        inflation: Multiplicative inflation factor.
        store_predicted_ensemble: Whether to store the incoming forecast ensemble.
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
