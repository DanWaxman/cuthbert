import itertools

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array, random

from cuthbert import filter, smoother
from cuthbert.gaussian import taylor
from cuthbert.gaussian.taylor.types import (
    GetDynamicsLogDensity,
    LogConditionalDensity,
    LogDensity,
    LogPotential,
)
from cuthbert.gaussian.types import (
    LinearizedKalmanFilterState,
)
from cuthbert.inference import Filter, Smoother
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.stats import multivariate_normal
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture
def dtype(request):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", request.param == jnp.float64)
    yield request.param
    jax.config.update("jax_enable_x64", previous)


def _load_taylor_init_and_dynamics(
    m0: Array,
    chol_P0: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
) -> tuple[tuple[LogDensity, Array], GetDynamicsLogDensity]:
    """Builds linearized log density Kalman filter and smoother objects and model_inputs
    for a linear-Gaussian SSM.
    """

    def init_log_density(x):
        return multivariate_normal.logpdf(x, m0, chol_P0)

    init_linearization_point = jnp.zeros_like(m0)

    def get_dynamics_log_density(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogConditionalDensity, Array, Array]:
        def dynamics_log_density(x_prev, x):
            return multivariate_normal.logpdf(
                x,
                Fs[model_inputs - 1] @ x_prev + cs[model_inputs - 1],
                chol_Qs[model_inputs - 1],
            )

        return (
            dynamics_log_density,
            jnp.zeros_like(m0),
            jnp.zeros_like(m0),
        )

    return (init_log_density, init_linearization_point), get_dynamics_log_density


def load_taylor_inference(
    m0: Array,
    chol_P0: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
    Hs: Array,
    ds: Array,
    chol_Rs: Array,
    ys: Array,
    associative_filter: bool = False,
    ignore_nan_dims: bool = False,
) -> tuple[Filter, Smoother, Array]:
    """Builds linearized log density Kalman filter and smoother objects and model_inputs
    for a linear-Gaussian SSM.
    """
    init_params, get_dynamics_log_density = _load_taylor_init_and_dynamics(
        m0, chol_P0, Fs, cs, chol_Qs
    )

    def get_observation_log_density(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogConditionalDensity, Array, Array]:
        def observation_log_density(x, y):
            return multivariate_normal.logpdf(
                y,
                Hs[model_inputs - 1] @ x + ds[model_inputs - 1],
                chol_Rs[model_inputs - 1],
            )

        return (
            observation_log_density,
            jnp.zeros_like(m0),
            ys[model_inputs - 1],
        )

    filter = taylor.build_filter(
        *init_params,
        get_dynamics_log_density,
        get_observation_log_density,
        associative=associative_filter,
        ignore_nan_dims=ignore_nan_dims,
    )
    smoother = taylor.build_smoother(
        get_dynamics_log_density,
        ignore_nan_dims=ignore_nan_dims,
        store_gain=True,
    )
    model_inputs = jnp.arange(len(ys) + 1)
    return filter, smoother, model_inputs


seeds = [0, 42, 99, 123, 456]
x_dims = [3]
y_dims = [1, 2]
num_time_steps = [1, 25]

common_params = list(itertools.product(seeds, x_dims, y_dims, num_time_steps))


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_offline_filter(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    if num_time_steps > 1:
        # Set an observation to nan
        ys = ys.at[1, 0].set(jnp.nan)

    taylor_filter, _, model_inputs = load_taylor_inference(
        m0,
        chol_P0,
        Fs,
        cs,
        chol_Qs,
        Hs,
        ds,
        chol_Rs,
        ys,
        associative_filter=False,
        ignore_nan_dims=True,
    )

    init_state = taylor_filter.init_prepare()

    # Run sequential sqrt filter
    seq_states = filter(taylor_filter, model_inputs[1:], init_state, parallel=False)
    seq_means, seq_chol_covs, seq_ells = (
        seq_states.mean,
        seq_states.chol_cov,
        seq_states.log_normalizing_constant,
    )

    associative_taylor_filter, _, model_inputs = load_taylor_inference(
        m0,
        chol_P0,
        Fs,
        cs,
        chol_Qs,
        Hs,
        ds,
        chol_Rs,
        ys,
        associative_filter=True,
        ignore_nan_dims=True,
    )

    associative_init_state = associative_taylor_filter.init_prepare()

    # Run associative filter with parallel=False§
    seq_ass_states = filter(
        associative_taylor_filter,
        model_inputs[1:],
        associative_init_state,
        parallel=False,
    )
    seq_ass_means, seq_ass_chol_covs, seq_ass_ells = (
        seq_ass_states.mean,
        seq_ass_states.chol_cov,
        seq_ass_states.log_normalizing_constant,
    )

    # Run associative filter with parallel=True
    par_ass_states = filter(
        associative_taylor_filter,
        model_inputs[1:],
        associative_init_state,
        parallel=True,
    )
    par_ass_means, par_ass_chol_covs, par_ass_ells = (
        par_ass_states.mean,
        par_ass_states.chol_cov,
        par_ass_states.log_normalizing_constant,
    )

    # Run the standard Kalman filter.
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ells = std_kalman_filter(
        m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    seq_ass_covs = seq_ass_chol_covs @ seq_ass_chol_covs.transpose(0, 2, 1)
    par_ass_covs = par_ass_chol_covs @ par_ass_chol_covs.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_ells),
        (seq_ass_means, seq_ass_covs, seq_ass_ells),
        (par_ass_means, par_ass_covs, par_ass_ells),
        (des_means, des_covs, des_ells),
        rtol=1e-5,
        atol=1e-8,
    )


@pytest.mark.parametrize("seed", [1, 43, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [1, 10])
@pytest.mark.parametrize("y_dim", [1, 5])
@pytest.mark.parametrize("associative", [True, False])
def test_filter_noop(seed, x_dim, y_dim, associative):
    m0, chol_P0 = generate_lgssm(seed, x_dim, y_dim, 0)[:2]

    def init_log_density(x):
        return multivariate_normal.logpdf(x, m0, chol_P0)

    init_linearization_point = jnp.zeros_like(m0)

    def get_noop_dynamics_log_density(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogConditionalDensity, Array, Array]:
        return (
            lambda x_prev, x: multivariate_normal.logpdf(x, x_prev, jnp.eye(x_dim)),
            jnp.full_like(m0, jnp.nan),
            jnp.full_like(m0, jnp.nan),
        )  # For Taylor we indicate noop dynamics by setting both linearization points to NaNs

    # Log density
    def get_noop_observation_log_density(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogConditionalDensity, Array, Array]:
        return (
            lambda x, y: multivariate_normal.logpdf(
                y,
                x,
                jnp.zeros((y_dim, y_dim)),
            ),
            jnp.zeros_like(m0),
            jnp.full(y_dim, jnp.nan),
        )

    filter_obj = taylor.build_filter(
        init_log_density,
        init_linearization_point,
        get_dynamics_log_density=get_noop_dynamics_log_density,
        get_observation_func=get_noop_observation_log_density,
        associative=associative,
    )

    state = filter_obj.init_prepare()
    prep_state = filter_obj.filter_prepare(None)
    filtered_state = filter_obj.filter_combine(state, prep_state)

    chex.assert_trees_all_close(
        (state.mean, state.chol_cov @ state.chol_cov.T, state.log_normalizing_constant),
        (
            filtered_state.mean,
            filtered_state.chol_cov @ filtered_state.chol_cov.T,
            filtered_state.log_normalizing_constant,
        ),
        rtol=1e-10,
    )  # Test covs rather than chol_covs because signs can be different

    # Log potential
    def get_noop_observation_log_potential(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogPotential, Array]:
        return (
            lambda x: multivariate_normal.logpdf(x, m0, chol_P0),
            jnp.full_like(m0, jnp.nan),
        )

    filter_log_potential_obj = taylor.build_filter(
        init_log_density,
        init_linearization_point,
        get_dynamics_log_density=get_noop_dynamics_log_density,
        get_observation_func=get_noop_observation_log_potential,
        associative=associative,
    )

    filtered_log_potential_state = filter_log_potential_obj.filter_combine(
        filter_log_potential_obj.init_prepare(),
        filter_log_potential_obj.filter_prepare(None),
    )

    chex.assert_trees_all_close(
        (state.mean, state.chol_cov @ state.chol_cov.T, state.log_normalizing_constant),
        (
            filtered_log_potential_state.mean,
            filtered_log_potential_state.chol_cov
            @ filtered_log_potential_state.chol_cov.T,
            filtered_log_potential_state.log_normalizing_constant,
        ),
        atol=1e-10,
        rtol=1e-10,
    )  # Test covs rather than chol_covs because signs can be different


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_smoother(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    log_density_filter, log_density_smoother, model_inputs = load_taylor_inference(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
    )

    # Run the Kalman filter and the standard Kalman smoother.
    init_state = log_density_filter.init_prepare()
    filt_states = filter(log_density_filter, model_inputs[1:], init_state)
    filt_means, filt_chol_covs = filt_states.mean, filt_states.chol_cov
    filt_covs = filt_chol_covs @ filt_chol_covs.transpose(0, 2, 1)
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    (des_means, des_covs), des_cross_covs = std_kalman_smoother(
        filt_means, filt_covs, Fs, cs, Qs
    )

    # Run the sequential and parallel versions of the square root smoother.
    seq_smoother_states = smoother(
        log_density_smoother, filt_states, model_inputs, parallel=False
    )
    seq_means, seq_chol_covs, seq_gains = (
        seq_smoother_states.mean,
        seq_smoother_states.chol_cov,
        seq_smoother_states.gain,
    )

    par_smoother_states = smoother(
        log_density_smoother, filt_states, model_inputs, parallel=True
    )
    par_means, par_chol_covs, par_gains = (
        par_smoother_states.mean,
        par_smoother_states.chol_cov,
        par_smoother_states.gain,
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    par_covs = par_chol_covs @ par_chol_covs.transpose(0, 2, 1)
    seq_cross_covs = seq_gains[:-1] @ seq_covs[1:]
    par_cross_covs = par_gains[:-1] @ par_covs[1:]
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_cross_covs),
        (par_means, par_covs, par_cross_covs),
        (des_means, des_covs, des_cross_covs),
        rtol=1e-5,
        atol=1e-8,
    )

    seq_default_mi = smoother(log_density_smoother, filt_states, parallel=False)
    chex.assert_trees_all_close(seq_default_mi, seq_smoother_states, rtol=1e-10)

    par_default_mi = smoother(log_density_smoother, filt_states, parallel=True)
    chex.assert_trees_all_close(par_default_mi, par_smoother_states, rtol=1e-10)


def load_taylor_inference_potential(
    m0: Array,
    chol_P0: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
    ms: Array,
    chol_Rs: Array,
    associative_filter: bool = False,
    ignore_nan_dims: bool = False,
) -> tuple[Filter, Smoother, Array]:
    """Builds linearized log density Kalman filter and smoother objects and model_inputs
    for a linear-Gaussian SSM.

    Uses Gaussian log potential for observations instead of conditional log density.
    """
    init_params, get_dynamics_log_density = _load_taylor_init_and_dynamics(
        m0, chol_P0, Fs, cs, chol_Qs
    )

    def get_observation_log_potential(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogPotential, Array]:
        def observation_log_potential(x):
            return multivariate_normal.logpdf(
                x, ms[model_inputs - 1], chol_Rs[model_inputs - 1]
            )

        return (
            observation_log_potential,
            jnp.zeros_like(m0),
        )

    filter = taylor.build_filter(
        *init_params,
        get_dynamics_log_density,
        get_observation_log_potential,
        associative=associative_filter,
        ignore_nan_dims=ignore_nan_dims,
    )
    smoother = taylor.build_smoother(
        get_dynamics_log_density, ignore_nan_dims=ignore_nan_dims
    )
    model_inputs = jnp.arange(len(ms) + 1)
    return filter, smoother, model_inputs


common_params_potential = list(itertools.product(seeds, x_dims, num_time_steps))


@pytest.mark.parametrize("seed,x_dim,num_time_steps", common_params_potential)
def test_offline_filter_potential(seed, x_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, _, _, chol_Rs, ms = generate_lgssm(
        seed, x_dim, x_dim, num_time_steps
    )

    taylor_filter, _, model_inputs = load_taylor_inference_potential(
        m0, chol_P0, Fs, cs, chol_Qs, ms, chol_Rs, associative_filter=False
    )

    init_state = taylor_filter.init_prepare()

    # Run sequential sqrt filter
    seq_states = filter(taylor_filter, model_inputs[1:], init_state, parallel=False)
    seq_means, seq_chol_covs, seq_ells = (
        seq_states.mean,
        seq_states.chol_cov,
        seq_states.log_normalizing_constant,
    )

    associative_taylor_filter, _, model_inputs = load_taylor_inference_potential(
        m0, chol_P0, Fs, cs, chol_Qs, ms, chol_Rs, associative_filter=True
    )

    associative_init_state = associative_taylor_filter.init_prepare()

    # Run associative filter with parallel=False
    seq_ass_states = filter(
        associative_taylor_filter,
        model_inputs[1:],
        associative_init_state,
        parallel=False,
    )
    seq_ass_means, seq_ass_chol_covs, seq_ass_ells = (
        seq_ass_states.mean,
        seq_ass_states.chol_cov,
        seq_ass_states.log_normalizing_constant,
    )

    par_ass_states = filter(
        associative_taylor_filter,
        model_inputs[1:],
        associative_init_state,
        parallel=True,
    )
    par_ass_means, par_ass_chol_covs, par_ass_ells = (
        par_ass_states.mean,
        par_ass_states.chol_cov,
        par_ass_states.log_normalizing_constant,
    )

    # Run the standard Kalman filter.
    Hs = jnp.eye(x_dim)[None].repeat(num_time_steps, axis=0)
    ds = jnp.zeros((num_time_steps, x_dim))
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ells = std_kalman_filter(
        m0, P0, Fs, cs, Qs, Hs, ds, Rs, ms
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    seq_ass_covs = seq_ass_chol_covs @ seq_ass_chol_covs.transpose(0, 2, 1)
    par_ass_covs = par_ass_chol_covs @ par_ass_chol_covs.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_ells),
        (seq_ass_means, seq_ass_covs, seq_ass_ells),
        (par_ass_means, par_ass_covs, par_ass_ells),
        (des_means, des_covs, des_ells),
        rtol=1e-5,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "associative,parallel", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("loading", [0.0, 1.0])
@pytest.mark.parametrize("log_rate", [-20.0, 0.6931471805599453])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], indirect=True)
def test_poisson_potential_evidence(associative, parallel, loading, log_rate, dtype):
    counts = jnp.array([2.0, 3.0, 7.0], dtype=dtype)
    anchor = jnp.zeros(1, dtype=dtype)
    variance = 0.25

    def prior(x):
        return jax.scipy.stats.norm.logpdf(x, scale=variance**0.5).sum()

    def dynamics(state, count):
        return lambda prev, x: prior(x), anchor, anchor

    def evidence(log_rate):
        def observation(state, count):
            def log_potential(x):
                eta = log_rate + loading * x[0]
                return count * eta - jnp.exp(eta) - jax.scipy.special.gammaln(count + 1)

            return log_potential, anchor

        alg = taylor.build_filter(
            prior,
            anchor,
            dynamics,
            observation,
            associative=associative,
            ignore_nan_dims=True,
        )
        states = filter(alg, counts, alg.init_prepare(), parallel=parallel)
        ell = states.log_normalizing_constant
        return ell[-1], jnp.diff(ell)

    def expected_increments(log_rate):
        rate = jnp.exp(log_rate)
        value = counts * log_rate - rate - jax.scipy.special.gammaln(counts + 1)
        grad = (counts - rate) * loading
        curvature = rate * loading**2
        return (
            value
            + 0.5 * variance * grad**2 / (1.0 + variance * curvature)
            - 0.5 * jnp.log1p(variance * curvature)
        )

    log_rate = jnp.asarray(log_rate, dtype=dtype)
    (total, increments), score = jax.jit(jax.value_and_grad(evidence, has_aux=True))(
        log_rate
    )
    expected = expected_increments(log_rate)
    expected_score = jax.grad(lambda b: expected_increments(b).sum())(log_rate)
    chex.assert_trees_all_close(
        (total, increments, score),
        (expected.sum(), expected, expected_score),
        rtol=2e-5 if dtype == jnp.float32 else 1e-7,
        atol=2e-5 if dtype == jnp.float32 else 1e-7,
    )


@pytest.mark.parametrize(
    "associative,parallel", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "dtype,bias",
    [(jnp.float32, -10.0), (jnp.float32, 0.6931471805599453), (jnp.float64, -20.0)],
    indirect=["dtype"],
)
def test_poisson_potential_evidence_correlated_dynamics(
    associative, parallel, dtype, bias
):
    # Nonzero dynamics and off-centre anchors also exercise associative composition.
    counts = jnp.array([1.0, 3.0, 2.0])
    anchors = jnp.array([0.2, -0.4, 0.1])

    def prior(x):
        return jax.scipy.stats.norm.logpdf(x, loc=0.7, scale=0.5**0.5).sum()

    def dynamics(state, inputs):
        def log_density(prev, x):
            return jax.scipy.stats.norm.logpdf(x, loc=0.8 * prev + 0.3, scale=0.5).sum()

        return log_density, jnp.zeros(1), jnp.zeros(1)

    def evidence(b):
        def observation(state, inputs):
            count, anchor = inputs

            def log_potential(x):
                eta = b + x[0]
                return count * eta - jnp.exp(eta) - jax.scipy.special.gammaln(count + 1)

            return log_potential, anchor[None]

        alg = taylor.build_filter(
            prior, jnp.zeros(1), dynamics, observation, associative=associative
        )
        states = filter(alg, (counts, anchors), alg.init_prepare(), parallel=parallel)
        return states.log_normalizing_constant[-1], (
            states.mean[:, 0],
            jnp.square(states.chol_cov[:, 0, 0]),
            states.log_normalizing_constant,
        )

    def expected(b):
        mean, variance, ell = 0.7, 0.5, 0.0
        means, variances, ells = (
            [jnp.asarray(mean)],
            [jnp.asarray(variance)],
            [jnp.asarray(ell)],
        )
        for count, anchor in zip(counts, anchors):
            mean, variance = 0.8 * mean + 0.3, 0.8**2 * variance + 0.25
            eta = b + anchor
            curvature = jnp.exp(eta)
            value = count * eta - curvature - jax.scipy.special.gammaln(count + 1)
            gradient = count - curvature
            residual = mean - anchor
            score = gradient - curvature * residual
            denominator = 1.0 + variance * curvature
            updated_variance = variance / denominator
            ell += (
                value
                + gradient * residual
                - 0.5 * curvature * residual**2
                + 0.5 * updated_variance * score**2
                - 0.5 * jnp.log1p(variance * curvature)
            )
            mean, variance = mean + updated_variance * score, updated_variance
            means.append(mean)
            variances.append(variance)
            ells.append(ell)
        return ell, tuple(jnp.stack(xs) for xs in (means, variances, ells))

    actual = jax.jit(jax.value_and_grad(evidence, has_aux=True))(jnp.asarray(bias))
    reference = jax.value_and_grad(expected, has_aux=True)(jnp.asarray(bias))
    chex.assert_trees_all_close(actual, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("active", [(0, 1, 2), (0, 1), (0, 2), (1, 2)])
def test_potential_evidence_active_coordinates(active):
    from cuthbert.gaussian.taylor.non_associative_filter import (
        _potential_log_evidence,
        process_observation,
    )

    active = jnp.array(active)
    mean = jnp.array([0.2, -0.7, 0.5])
    chol_cov = jnp.array([[1.0, 0.0, 0.0], [0.4, 0.8, 0.0], [-0.2, 0.3, 0.7]])
    target = jnp.array([0.1, 0.8, -0.6])[active]
    covariance = jnp.array([[0.7, 0.1, 0.2], [0.1, 1.3, -0.1], [0.2, -0.1, 0.9]])
    covariance = covariance[active[:, None], active]
    anchor = jnp.array([-0.3, 0.2, 0.6])

    def evidence(offset):
        def potential(x):
            return offset + jax.scipy.stats.multivariate_normal.logpdf(
                x[active], target, covariance
            )

        _, d, chol_R, observation, metadata = process_observation(
            (potential, anchor), ignore_nan_dims=True
        )
        return _potential_log_evidence(
            mean, chol_cov, d, chol_R, observation, metadata, jnp.asarray(-123.0)
        )

    offset = jnp.asarray(1.7)
    prior_covariance = chol_cov @ chol_cov.T
    reference = offset + jax.scipy.stats.multivariate_normal.logpdf(
        target, mean[active], covariance + prior_covariance[active[:, None], active]
    )
    actual, gradient = jax.jit(jax.value_and_grad(evidence))(offset)
    chex.assert_trees_all_close((actual, gradient), (reference, jnp.asarray(1.0)))


def test_potential_evidence_singular_fallback():
    from cuthbert.gaussian.taylor.non_associative_filter import (
        _potential_log_evidence,
        _TaylorPotential,
    )

    def evidence(value):
        return _potential_log_evidence(
            jnp.zeros(2),
            jnp.eye(2),
            jnp.ones(2),
            jnp.diag(jnp.array([1.0, 0.0])),
            jnp.zeros(2),
            _TaylorPotential(jnp.zeros(2), value),
            jnp.asarray(-3.0),
        )

    actual, gradient = jax.jit(jax.value_and_grad(evidence))(jnp.asarray(1.0))
    chex.assert_trees_all_close(
        (actual, gradient), (jnp.asarray(-3.0), jnp.asarray(0.0))
    )


num_factors = [3, 10]
common_params_factorial_init = list(itertools.product(seeds, x_dims, num_factors))


@pytest.mark.parametrize("seed,x_dim,num_factors", common_params_factorial_init)
def test_factorial_init(seed, x_dim, num_factors):
    key = random.PRNGKey(seed)
    mean_key, chol_cov_key, lin_p_key = random.split(key, 3)

    m0 = random.normal(mean_key, shape=(x_dim,))
    sqrt_P0 = random.normal(chol_cov_key, shape=(x_dim, x_dim))
    P0 = sqrt_P0 @ sqrt_P0.T
    chol_P0 = jnp.linalg.cholesky(P0)
    lin_p = random.normal(lin_p_key, shape=(num_factors, x_dim))

    def init_log_density(x):
        factor_log_densities = jax.vmap(
            lambda factor_x: multivariate_normal.logpdf(factor_x, m0, chol_P0)
        )(x)
        return jnp.sum(factor_log_densities)

    init_linearization_point = lin_p

    init_state = taylor.non_associative_filter.init_prepare(
        init_log_density, init_linearization_point
    )
    init_state_assoc = taylor.associative_filter.init_prepare(
        init_log_density, init_linearization_point
    )

    factorial_mean = jnp.tile(m0[None], (num_factors, 1))
    factorial_P0 = jnp.tile(P0[None], (num_factors, 1, 1))

    chex.assert_trees_all_close(
        (factorial_mean, factorial_P0, 0.0),
        (
            init_state.mean,
            init_state.chol_cov @ init_state.chol_cov.transpose(0, 2, 1),
            init_state.log_normalizing_constant,
        ),
        (
            init_state_assoc.mean,
            init_state_assoc.chol_cov @ init_state_assoc.chol_cov.transpose(0, 2, 1),
            init_state_assoc.log_normalizing_constant,
        ),
        rtol=1e-5,
        atol=1e-8,
    )
