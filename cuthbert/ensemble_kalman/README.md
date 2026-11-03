# Ensemble Kalman

The core atomic functions are in [`cuthbertlib.ensemble_kalman.filtering`](../../cuthbertlib/ensemble_kalman/filtering.py)
and [`cuthbertlib.ensemble_kalman.smoothing`](../../cuthbertlib/ensemble_kalman/smoothing.py). The
high-level interfaces are in
[`cuthbert.ensemble_kalman.ensemble_kalman_filter`](ensemble_kalman_filter.py),
[`cuthbert.ensemble_kalman.local_ensemble_transform_kalman_filter`](local_ensemble_transform_kalman_filter.py), and
[`cuthbert.ensemble_kalman.ensemble_rts_smoother`](ensemble_rts_smoother.py).

## Ensemble Kalman filter

<!-- --8<-- [start:enkf] -->
The EnKF treats the filtering distribution as **Gaussian**, but represents it with an ensemble of $N$ members $x^{(i)}$ instead of storing a mean and covariance and linearizing $f$ or $h$. The implied mean and covariance are the usual sample mean and sample covariance of the members.

**Predict.** Each member is advanced with the dynamics and process noise. **Multiplicative inflation** (optional) rescales deviations from the new ensemble mean by a factor $(1+\delta)$ to combat underspread ensembles.

**Update.** From deviations in state and observation space, form empirical cross-covariance $C_{xy}$ and innovation covariance $S$ in observation space (including observation noise). The Kalman gain $K \approx C_{xy} S^{-1}$ gives a Kalman-like correction to each member (e.g. stochastic EnKF with random observation perturbations).

The EnKF allows for storing its predicted states, $x_{t \mid t - 1}$, through the `store_predicted_ensemble` flag. This flag is required when the filtering outputs are to be used by an EnRTS smoother.

**Large observation dimensions.** Setting `ensemble_subspace=True` performs the analysis in the $N$-dimensional ensemble subspace via the Woodbury identity. It is algebraically exact and cheaper whenever $N$ is much smaller than the observation dimension. This cost can be further reduced by passing scalar or diagonal `chol_R`.  It is incompatible with both localization callbacks below. See the [`cuthbertlib` filtering documentation](../../api_cuthbertlib/ensemble_kalman/filtering.md) for the details.

See Algorithm 2 in Appendix A in [Calvello, Reich, and Stuart., Ensemble Kalman Methods: A Mean Field Perspective](https://arxiv.org/abs/2209.11371) for the EnKF algorithm which accomodates non-linear observation functions $h$. Note that this algorithm corresponds to the `perturbed_obs = True (Default)` option in the EnKF implementation. This boolean flag is represented by `s` in Algorithm 10.2 of [Sanz-Alonso et al., *Inverse Problems and Data Assimilation*](https://arxiv.org/abs/1810.06191), which was only written for linear $h$.
<!-- --8<-- [end:enkf] -->

## Local ensemble transform Kalman filter

<!-- --8<-- [start:letkf] -->
The Local Ensemble Transform Kalman Filter (LETKF) propagates an ensemble using
the same simulator and inflation convention as the EnKF, then applies a
deterministic ensemble-space square-root analysis at each state coordinate.
It uses the symmetric transform described by
[Hunt, Kostelich, and Szunyogh (2007)](https://doi.org/10.1016/j.physd.2006.11.008).

`local_ensemble_transform_kalman_filter.build_filter` accepts `init_sample`,
`get_dynamics`, `get_observations`, and `n_particles` as in the EnKF, plus a
`get_localization(model_inputs)` callback. This callback returns observation
indices and precision weights, both shaped `(x_dim, max_local_observations)`.
Each row defines the observations affecting one state coordinate. Indices refer
to the original observation order; use `-1` for padding. Nonpadding indices must
be in range and unique within a row. Weights must be finite and nonnegative;
padding weights are ignored and zero weights exclude observations. These value
constraints are caller requirements. Shapes remain fixed under JIT, while
neighborhood indices and weights may vary over time. The existing `gaspari_cohn`
and `gaussian` utilities can construct weights from local distances.

The observation callback must return a scalar `chol_R` or a 1D diagonal factor of
length `y_dim`; dense factors, including dense diagonal matrices, are not
supported. Observation noise covariance is `chol_R ** 2` on the diagonal.
NaNs in observations are excluded from both the analysis and the likelihood.
Coordinates with no active observations preserve their forecast members.
This applies to the default per-coordinate analysis; weight interpolation can
carry analysis information from nearby coarse points into data-void regions.

The returned filter uses `EnKFState`, explicit initialization and random keys,
and sequential `cuthbert.filter(..., parallel=False)` execution. Forecast
deviations are multiplied by `1 + inflation`. As with the EnKF,
`store_predicted_ensemble=True` retains the incoming forecast in each state.
Set `batch_size` on the builder to analyze that many state coordinates in
parallel within each `jax.lax.map` batch. The default `None` processes one
coordinate at a time; larger batches use more memory. A final partial batch is
handled automatically.

**Optional weight interpolation.** Supply `get_weight_interpolation(model_inputs)`
to enable the method of [Yang et al. (2009)](https://doi.org/10.1002/qj.353).
With this option, `get_localization` returns neighborhoods on a coarse analysis
grid, shaped `(n_analysis, max_local_observations)`, rather than one neighborhood
per state coordinate. The interpolation callback returns integer coarse-analysis
indices and interpolation coefficients, both shaped `(x_dim, stencil_size)`.
Indices refer to rows of `get_localization`; use `-1` for padding. Nonpadding
indices must be in range, and finite nonnegative coefficients must sum to one
over nonpadding entries in each row. Padding coefficients are ignored.
Shapes stay fixed under JIT; these value constraints remain caller requirements.

The caller supplies the interpolation geometry, for example two-point linear
or four-point bilinear stencils. Both the mean-analysis weights and symmetric
ensemble transforms are interpolated, then applied to each fine coordinate's
own forecast anomalies. State fields and spatial analysis increments are not
interpolated. Only the coarse points require ensemble-space factorizations;
their transforms are retained while fine coordinates are updated in batches.
`get_weight_interpolation=None` (the default) keeps the per-coordinate analysis.

**Likelihood.** `log_normalizing_constant` accumulates the global Gaussian
predictive log density of the full forecast observation ensemble, evaluated
before each local analysis. The observation covariance is the sample forecast
covariance plus the original observation-noise covariance. Each observation is
scored once; neighborhood scores are never summed. Localization affects later
forecasts, but neither localization nor weight interpolation directly changes
the current forecast's score. This is a
normalized predictive density approximation, not an unbiased estimate of the
original model likelihood. An entirely missing observation contributes zero.
<!-- --8<-- [end:letkf] -->

## Ensemble Rauch-Tung-Striebel smoother

<!-- --8<-- [start:enks] -->
The EnRTS is an ensemble RTS-like smoothing algorithm, similar in form to the RTS smoothing algorithm. It forms a backwards recursion using a gain $J_t$:

$$
x_{t \mid T}^{(i)} = x_{t \mid t}^{(i)} + J_t \left(x_{t+1 \mid T}^{(i)} - x_{t+1 \mid t}^{(i)}\right),
$$

where $J_t$ is determined by ensembles to be an empirical version of the Kalman gain.

The computation of $J_t$ involves the ensemble of $x_{t+1 \mid t}$; to avoid duplicate computation, the EnRTS therefore requires `store_predicted_ensemble=True` in the forward filtering run.

See [Raanes (2016)](https://doi.org/10.1002/qj.2728) for more info on the EnRTS algorithm and its equivalence to the ensemble Kalman smoother (EnKS), a simliar ensemble smoothing algorithm which consists of a forward pass of increasing dimension.

<!-- --8<-- [end:enks] -->

## Covariance localization

Cuthbert implements localization through two independent callbacks. `modify_cross_covariance` receives the empirical state-observation cross-covariance and the current model inputs, then returns the modified cross-covariance. The optional `construct_chol_innovation_covariance` callback instead receives normalized observation deviations $Y$, the observation-noise factor `chol_R`, and the current model inputs, then returns a valid generalized Cholesky factor of the complete localized innovation covariance $S=C_{yy}+R$. Both callbacks receive quantities in the original observation order. By default, `modify_cross_covariance` is `no_covariance_modifier`, which passes the covariance through unchanged, and `construct_chol_innovation_covariance` is `None` (which avoids a more costly QR solve).

Cuthbert is agnostic to the exact form of the cross-covariance modification. One common form is covariance tapering, which forms the modified covariance as an elementwise product with a tapering matrix. We provide a few common tapers in `cuthbertlib.ensemble_kalman.localization.py`.

For example, cross-covariance localization with the Gaspari-Cohn correlation function can be specified as follows:

```python
from cuthbertlib.ensemble_kalman.localization import gaspari_cohn


def modify_cross_covariance(cross_covariance, model_inputs):
    x = model_inputs.state_locations
    y = model_inputs.observation_locations
    taper = gaspari_cohn(
        x[:, None] - y[None, :],
        model_inputs.support_radius,
    )
    return taper * cross_covariance
```

When `construct_chol_innovation_covariance` is `None`, the EnKF uses the compact square-root construction `tria([Y, chol_R])`, whose input has $N+y_{\rm dim}$ columns. Cross-only localization retains this path. For observation-space tapering, `construct_tapered_chol_innovation_covariance` is one constructor option: it implements a square-root identity without explicitly forming $C_{yy}$ or directly factorizing $S$. The observation taper must be positive semidefinite and supplied through a valid factor `chol_taper`. This tapered construction expands the anomaly factor to $y_{\rm dim}N$ columns, so the `None` path remains preferable when marginal localization is not needed.

For gradient-based optimization of the localization function, the Gaspari-Cohn taper may be suboptimal, as its gradients are dead for points outside the support radius. In this case, a non-compact covariance taper such as a Gaussian kernel may be preferable.

```python
import jax.numpy as jnp

from cuthbertlib.ensemble_kalman import (
    construct_tapered_chol_innovation_covariance,
    gaussian,
)


def modify_cross_covariance(cross_covariance, model_inputs):
    x = model_inputs.state_locations
    y = model_inputs.observation_locations
    length_scale = jnp.exp(model_inputs.log_length_scale)
    taper = gaussian(x[:, None] - y[None, :], length_scale)
    return taper * cross_covariance


def construct_chol_innovation_covariance(Y, chol_R, model_inputs):
    y = model_inputs.observation_locations
    length_scale = jnp.exp(model_inputs.log_length_scale)
    taper = gaussian(y[:, None] - y[None, :], length_scale)
    chol_taper = jnp.linalg.cholesky(taper)
    return construct_tapered_chol_innovation_covariance(Y, chol_taper, chol_R)
```

Here, `Y` has already been divided by $\sqrt{N-1}$. If the taper is fixed over filtering steps, compute `chol_taper` once outside the callback and close over it instead.
