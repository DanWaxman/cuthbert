# Ensemble Kalman

The high-level interfaces built on these functions are in
[`cuthbert.ensemble_kalman.ensemble_kalman_filter`](../../cuthbert/ensemble_kalman/ensemble_kalman_filter.py),
[`cuthbert.ensemble_kalman.ensemble_transform_kalman_filter`](../../cuthbert/ensemble_kalman/ensemble_transform_kalman_filter.py), and
[`cuthbert.ensemble_kalman.ensemble_rts_smoother`](../../cuthbert/ensemble_kalman/ensemble_rts_smoother.py).

<!-- --8<-- [start:overview] -->
This sub-repository provides modular functions for ensemble Kalman methods, including the ensemble Kalman filter (EnKF) and ensemble RTS smoother (EnRTS smoother).

The core functions are:

- `predict`: Propagate ensemble members through nonlinear dynamics.
- `filter_update`: Update ensemble members with an observation using the EnKF update equation.
- `etkf_update`: Apply a global or local ensemble transform and compute a global Gaussian predictive log density.
- `smoother_update`: Apply one Ensemble Rauch-Tung-Striebel smoother update.
- `construct_tapered_chol_innovation_covariance`: Construct a tapered innovation covariance factor without forming the covariance densely.
- `gaspari_cohn`: Construct covariance tapers with the Gaspari-Cohn correlation function.
- `gaussian`: Construct smooth, non-compact covariance tapers with the Gaussian correlation function.
<!-- --8<-- [end:overview] -->

## Ensemble Kalman filtering

<!-- --8<-- [start:filtering] -->
Together, `predict` and `filter_update` can be used to perform an online EnKF filtering step.

The EnKF uses an ensemble of particles with a Kalman-style measurement update based on empirical covariances. Unlike the EKF, it does not require Jacobians, while naturally handling nonlinear dynamics.

### Large observation dimensions

By default, `filter_update` forms the empirical cross-covariance $C_{xy}$ and a generalized Cholesky factor of the innovation covariance $S = C_{yy} + R$, costing $\mathcal{O}(y_{\rm dim}^3 + N y_{\rm dim} x_{\rm dim})$ and storing arrays of size $y_{\rm dim}^2$ and $x_{\rm dim}y_{\rm dim}$.

Passing `ensemble_subspace=True` instead carries out the analysis in the $N$-dimensional subspace spanned by the ensemble, using the Woodbury identity. The update becomes $X C^{-1} Y^\intercal R^{-1}\delta$ with $C = I_N + Y^\intercal R^{-1} Y$, so the only factorization is $N \times N$ and neither $C_{xy}$, $S$, nor the Kalman gain is ever formed. The cost is $\mathcal{O}(N^2 x_{\rm dim} + N^2 y_{\rm dim} + N^3)$ plus the cost of applying $R^{-1}$. This is algebraically exact and is preferable whenever $N \ll y_{\rm dim}$; for $y_{\rm dim} \lesssim N$ the default path is cheaper. 

When $R^{-1}$ is applied by a dense Cholesky factor, this incurs a cost of $\mathcal{O}(Nd_y^2)$. One can reduce this to $\mathcal{O}(Nd_y)$ by passing a structured `chol_R`: a scalar for $\sigma^2 I$, or a 1D array of length $y_{\rm dim}$ for a diagonal factor. Note that the default path always requires a 2D `chol_R`.

Both localization hooks below are rejected with `ensemble_subspace=True`, as the Woodbury identity is inapplicable with tapering.

<!-- --8<-- [end:filtering] -->

## Ensemble transform filtering

<!-- --8<-- [start:etkf] -->
The predict step for the ETKF is the same as the EnKF, and the filters differ only in their `update` step; for the ETKF, we provide the `etkf_update`. By default, this performs a global ensemble transform update, and applies localized analyses if `observation_indices` and `localization_weights` are supplied. 

The global ensemble transform computes the symmetric square root factor $C^{-1/2}$, where $C = I_N + Y^\top R^{-1}Y$ and $Y$ is the matrix of centered forecast observation deviations divided by $\sqrt{N-1}$. The implementation uses `tria` to obtain a triangular factor $L$ of $C$ from the whitened deviations and the identity. The SVD $L = U\Sigma V^\top$ then gives $C^{-1/2} = U\Sigma^{-1}U^\top$. The update right-multiplies the forecast state deviations by $C^{-1/2}$ and shifts the ensemble mean by $XC^{-1}Y^\top R^{-1}(y-\bar h)$, where $X$ is the matrix of centered forecast state deviations divided by $\sqrt{N-1}$.

Indices and weights have shape `(x_dim, max_local_observations)` to determine local observations to be used in analysis. Indices use the original observation order (i.e., before pivoting for missingness). To pad (when `|local_observations| < max_local_observations`), one may pad with `-1` values. Weights must be finite and nonnegative, and non-padded indices must be valid and unique. 

For localized analysis, batching is supported via the `batch_size` parameter, which is used to `jax.lax.map` over analyses. The default is `None`, which does no batching.

Each update returns one global Gaussian predictive log density $\log\mathcal N(y;\bar h,R+YY^\top)$, where $y$ is the observation vector, $\bar h$ is the forecast observation ensemble mean, and $R$ is the original observation-noise covariance. All terms are restricted to observed dimensions. The high-level filter sums these scores over time to approximate the log marginal likelihood.

<!-- --8<-- [end:etkf] -->

## Ensemble Rauch-Tung-Striebel smoothing

<!-- --8<-- [start:smoothing] -->
The EnRTS algorithm provides a smoothing counterpart to the EnKF, based on the RTS smoother. It makes an empirical approximation of the RTS smoother gain, which is applied to the ensemble in a backwards pass to obtain a smoothing distribution. Note that, based on the outputs of the EnKF, the EnRTS step is entirely deterministic.
<!-- --8<-- [end:smoothing] -->

## Covariance localization

<!-- --8<-- [start:localization] -->
`filter_update` accepts two independent low-level hooks. `cross_covariance_modifier` receives one empirical state-observation cross-covariance and returns its modified value; it defaults to `no_covariance_modifier`. For observation-space covariance localization, one may optionally use `construct_chol_innovation_covariance`, which receives normalized observation deviations `Y` and a Cholesky factor of the observation noise covariance, `chol_R`, and returns a generalized Cholesky factor of the localized innovation covariance $C_{yy} + R$. All localization happens before handling of missing data, so localization happens in the original coordinates.

The most common form of localization is covariance tapering; cuthbertlib provides `gaspari_cohn` and `gaussian` correlation functions for this purpose. The Gaspari-Cohn correlation function is more classical, but has compact support which may cause difficulties in optimizing localization hyperparameters. The `gaussian` correlation function has infinite support, and may therefore have better gradient properties. For obsservation-space tapering, `cuthbertlib` provides a convenience `construct_tapered_chol_innovation_covariance`, which uses a Cholesky factor of the taper to construct a factor of the tapered innovation covariance. This comes at slightly higher cost, due to a larger QR solve.

<!-- --8<-- [end:localization] -->
