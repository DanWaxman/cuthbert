import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

from cuthbert import filter
from cuthbert.ensemble_kalman import local_ensemble_transform_kalman_filter as letkf
from cuthbertlib.ensemble_kalman import gaussian
from tests.cuthbertlib.kalman.test_filtering import std_update


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_letkf_inference(
    store_predicted_ensemble=False,
    log_length_scale=None,
    batch_size=None,
    interpolate_weights=False,
):
    F = jnp.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.1], [0.1, 0.0, 0.85]])
    c = jnp.array([0.05, -0.02, 0.03])
    H = jnp.array([[1.0, 0.0, 0.2], [0.0, 0.4, 1.0]])
    chol_R = jnp.array([0.3, 0.5])
    ys = jnp.array([[0.1, -0.2], [0.3, 0.1], [-0.1, 0.4]])
    indices = jnp.broadcast_to(jnp.arange(2), (3, 2))
    weights = jnp.ones((3, 2))
    if log_length_scale is not None:
        distances = jnp.array([[0.0, 0.8], [0.5, 0.3], [1.0, 0.2]])
        weights = gaussian(distances, jnp.exp(log_length_scale))
        # This coordinate has an identity transform, while the others have
        # rank-deficient local anomaly matrices (two observations, eight members).
        indices = indices.at[2].set(-1)

    get_weight_interpolation = None
    if interpolate_weights:
        indices, weights = indices[jnp.array([0, 2])], weights[jnp.array([0, 2])]
        fraction = 0.5 if log_length_scale is None else jax.nn.sigmoid(log_length_scale)
        interpolation_indices = jnp.array([[0, -1], [0, 1], [1, -1]])
        interpolation_weights = jnp.array(
            [[1.0, 0.0], [1 - fraction, fraction], [1.0, 0.0]]
        )

        def get_weight_interpolation(model_inputs):
            return interpolation_indices, interpolation_weights

    inference = letkf.build_filter(
        init_sample=lambda key: 0.2 + 0.6 * random.normal(key, (3,)),
        get_dynamics=lambda _: lambda x, key: F @ x + c,
        get_observations=lambda y: (lambda x: H @ x, chol_R, y),
        n_particles=8,
        get_localization=lambda _: (indices, weights),
        inflation=0.05,
        store_predicted_ensemble=store_predicted_ensemble,
        batch_size=batch_size,
        get_weight_interpolation=get_weight_interpolation,
    )
    return inference, ys, F, c, H, chol_R


class Test(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(store_predicted_ensemble=[False, True], batch_size=[None, 2])
    def test_filter(self, store_predicted_ensemble, batch_size):
        inference, ys, F, c, H, chol_R = load_letkf_inference(
            store_predicted_ensemble, batch_size=batch_size
        )
        init_key, filter_key = random.split(random.key(2))
        initial = inference.init_prepare(key=init_key)
        assert not inference.associative
        states = self.variant(filter, static_argnames=("filter_obj",))(
            inference, ys, initial, key=filter_key
        )

        mean, cov = initial.mean, jnp.cov(initial.ensemble.T)
        means, covs, log_likelihoods = [mean], [cov], [jnp.array(0.0)]
        for y in ys:
            mean, cov, ll = std_update(
                F @ mean + c,
                1.05**2 * (F @ cov @ F.T),
                H,
                jnp.zeros(2),
                jnp.diag(chol_R**2),
                y,
            )
            means.append(mean)
            covs.append(cov)
            log_likelihoods.append(log_likelihoods[-1] + ll)
        chex.assert_trees_all_close(
            (
                states.mean,
                states.chol_cov @ states.chol_cov.transpose(0, 2, 1),
                states.log_normalizing_constant,
            ),
            (jnp.stack(means), jnp.stack(covs), jnp.stack(log_likelihoods)),
            rtol=1e-10,
            atol=1e-10,
        )
        if store_predicted_ensemble:
            forecasts = states.ensemble[:-1] @ F.T + c
            forecast_means = forecasts.mean(axis=1, keepdims=True)
            expected = forecast_means + 1.05 * (forecasts - forecast_means)
            chex.assert_trees_all_close(
                states.predicted_ensemble[1:], expected, atol=1e-12
            )
        else:
            assert states.predicted_ensemble is None


@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("interpolate_weights", [False, True])
def test_localized_log_likelihood_gradient(batch_size, interpolate_weights):
    init_key, filter_key = random.split(random.key(3))

    @jax.jit
    def log_likelihood(log_length_scale):
        inference, ys, *_ = load_letkf_inference(
            log_length_scale=log_length_scale,
            batch_size=batch_size,
            interpolate_weights=interpolate_weights,
        )
        initial = inference.init_prepare(key=init_key)
        return filter(inference, ys, initial, key=filter_key).log_normalizing_constant[
            -1
        ]

    log_length_scale = jnp.log(0.7)
    value, gradient = jax.value_and_grad(log_likelihood)(log_length_scale)
    epsilon = 1e-4
    difference = (
        log_likelihood(log_length_scale + epsilon)
        - log_likelihood(log_length_scale - epsilon)
    ) / (2 * epsilon)
    assert jnp.isfinite(value)
    assert jnp.abs(gradient) > 1e-4
    chex.assert_trees_all_close(gradient, difference, rtol=1e-5, atol=1e-7)
