import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbertlib.ensemble_kalman import letkf_update
from cuthbertlib.ensemble_kalman.letkf import _symmetric_inverse_sqrt
from tests.cuthbertlib.kalman.test_filtering import std_update


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("batch_size", [None, 2, 4])
def test_update_matches_kalman(diagonal, batch_size):
    ensemble = random.normal(random.key(0), (8, 3))
    H = jnp.array([[1.0, 0.2, 0.0], [0.0, 0.5, 1.0]])
    offset = jnp.array([0.1, -0.2])
    y = jnp.array([0.4, -0.7])
    chol_R = jnp.array([0.4, 0.7]) if diagonal else jnp.array(0.4)
    indices = jnp.broadcast_to(jnp.arange(2), (3, 2))

    updated, ll = jax.jit(
        letkf_update, static_argnames=("observation_fn", "batch_size")
    )(
        ensemble,
        lambda x: H @ x + offset,
        chol_R,
        y,
        indices,
        jnp.ones((3, 2)),
        batch_size=batch_size,
    )
    expected_mean, expected_cov, expected_ll = std_update(
        ensemble.mean(0),
        jnp.cov(ensemble.T),
        H,
        offset,
        jnp.diag(jnp.broadcast_to(chol_R, y.shape) ** 2),
        y,
    )
    chex.assert_trees_all_close(
        (updated.mean(0), jnp.cov(updated.T), ll),
        (expected_mean, expected_cov, expected_ll),
        rtol=1e-10,
        atol=1e-10,
    )


def test_localization_and_missing_observations():
    ensemble = random.normal(random.key(1), (8, 3))
    y = jnp.array([0.5, jnp.nan, -0.8])
    chol_R = jnp.array([0.4, jnp.nan, 0.7])
    indices = jnp.array([[0, 2, -1, 1], [1, 2, -1, 0], [2, 0, -1, 1]])
    weights = jnp.array([[1.0, 0.0, 1.0, 0.0]] * 2 + [[0.0, 0.0, 1.0, 0.0]])

    def observation_fn(x):
        return x.at[1].set(jnp.nan)

    update = jax.jit(
        lambda y, indices, weights: letkf_update(
            ensemble, observation_fn, chol_R, y, indices, weights
        )
    )
    updated, ll = update(y, indices, weights)
    mean, cov = ensemble.mean(0), jnp.cov(ensemble.T)
    local_mean, local_cov, _ = std_update(
        mean, cov, jnp.eye(3)[:1], jnp.zeros(1), jnp.array([[0.4**2]]), y[:1]
    )
    chex.assert_trees_all_close(
        (updated[:, 0].mean(), jnp.var(updated[:, 0], ddof=1)),
        (local_mean[0], local_cov[0, 0]),
        atol=1e-10,
    )
    chex.assert_trees_all_equal(updated[:, 1:], ensemble[:, 1:])
    _, _, expected_ll = std_update(
        mean, cov, jnp.eye(3), jnp.zeros(3), jnp.diag(chol_R**2), y
    )
    chex.assert_trees_all_close(ll, expected_ll, atol=1e-10)

    # An observation outside every active neighborhood still affects the global score.
    changed, changed_ll = update(y.at[2].add(4), indices, weights)
    chex.assert_trees_all_equal(changed, updated)
    assert not jnp.isclose(changed_ll, ll)
    reweighted, reweighted_ll = update(y, indices, weights.at[0, 0].set(0.3))
    weighted_mean, weighted_cov, _ = std_update(
        mean, cov, jnp.eye(3)[:1], jnp.zeros(1), jnp.array([[0.4**2 / 0.3]]), y[:1]
    )
    chex.assert_trees_all_close(
        (reweighted[:, 0].mean(), jnp.var(reweighted[:, 0], ddof=1)),
        (weighted_mean[0], weighted_cov[0, 0]),
        atol=1e-10,
    )
    chex.assert_trees_all_equal(reweighted_ll, ll)

    empty, empty_ll = update(y, jnp.empty((3, 0), dtype=int), jnp.empty((3, 0)))
    chex.assert_trees_all_equal((empty, empty_ll), (ensemble, ll))
    missing, missing_ll = update(jnp.full_like(y, jnp.nan), indices, weights)
    chex.assert_trees_all_equal((missing, missing_ll), (ensemble, jnp.array(0.0)))


@pytest.mark.parametrize("rank_one", [False, True])
def test_inverse_sqrt_gradient_at_repeated_eigenvalues(rank_one):
    direction = jnp.array([[0.2, 0.1, -0.3], [0.1, -0.1, 0.2], [-0.3, 0.2, 0.4]])
    vector = jnp.array([0.3, -0.2, 0.6])
    matrix = jnp.eye(3) + (jnp.outer(vector, vector) if rank_one else 0)
    value, tangent = jax.jit(
        lambda matrix: jax.jvp(_symmetric_inverse_sqrt, (matrix,), (direction,))
    )(matrix)
    epsilon = 1e-5
    difference = (
        _symmetric_inverse_sqrt(matrix + epsilon * direction)
        - _symmetric_inverse_sqrt(matrix - epsilon * direction)
    ) / (2 * epsilon)
    chex.assert_trees_all_close(value @ matrix @ value, jnp.eye(3), atol=1e-12)
    chex.assert_trees_all_close(tangent, difference, rtol=1e-7, atol=1e-9)
    if not rank_one:
        chex.assert_trees_all_close(tangent, -0.5 * direction, atol=1e-12)
