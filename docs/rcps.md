# Risk-controlling prediction sets

The RCPS calibrators construct nested prediction intervals to control expected
coverage loss with high probability over the draw of the calibration set. The
default implements the bounded-loss Hoeffding upper confidence bound described
in [Bates et al. (2021)](https://arxiv.org/abs/2101.02703).
`ScaledIntervalRCPS` directly implements the asymmetric interval family used for
image-to-image regression by
[Angelopoulos et al. (2022)](https://arxiv.org/abs/2202.05265), while
`EnsembleRCPS` adapts that family to raw ensemble forecasts.

## Ensemble predictions

`EnsembleRCPS` is the primary interface for raw ensemble forecasts. Predictions
use `TensorBNCM`:

```text
(batch, *optional_dims, channel, ensemble)
```

By default, it uses the ensemble mean together with empirical ensemble
quantiles to construct

$$
T_{\lambda,\alpha}(X) =
[\bar f(X)-\lambda(\bar f(X)-q_{\alpha/2}(X)),
 \bar f(X)+\lambda(q_{1-\alpha/2}(X)-\bar f(X))].
$$

The two widths are scaled separately, so an asymmetric predictive distribution
produces an asymmetric interval around the ensemble mean. Widths are clamped by
`min_scale` to keep the family nested when an empirical quantile crosses the
mean. `mode="std"` is also available when a symmetric
mean-plus-or-minus-standard-deviation family is preferred. Both modes produce
nested intervals as lambda increases.

Float16 and bfloat16 ensembles are promoted to float32 for ensemble statistics
and output intervals so the positive width floor remains representable.

Calibration and prediction ensembles must use the same member count and
generation procedure so that their losses remain exchangeable.

```python
from autouq.calibrators import EnsembleRCPS

calibrator = EnsembleRCPS(
    alphas=[0.1, 0.2],
    deltas=[0.05, 0.1],
)
calibrator.calibrate(true_cal, ensemble_cal)

intervals = calibrator.predict(ensemble_test)
```

### Reusing ensemble statistics and memory cost

For a fixed input ensemble and alpha, changing lambda only rescales the interval
half-widths, and changing delta only changes the risk bound used to choose lambda.
The ensemble statistics therefore stay constant throughout the scale search and
across deltas. Recomputing the quantiles at every search step would repeatedly
sort the same ensemble values.

With the default `cache_statistics=True`, each `calibrate` or `predict` call
temporarily caches these three tensors:

- The ensemble mean (`centre`).
- The lower half-width, `max(centre - lower_quantile, min_scale)`.
- The upper half-width, `max(upper_quantile - centre, min_scale)`.

Each tensor has shape `(batch, *optional_dims, channel)`: the ensemble member
axis has been reduced. Both methods loop over alphas on the outside and deltas
on the inside: they finish every delta for the current alpha before advancing.
Quantile mode therefore keeps only the current alpha's tuple and releases it
before computing the next alpha's statistics. In `mode="std"`, all alphas share
one mean tensor and one clamped population-standard-deviation tensor; both
widths refer to that same standard-deviation tensor.

This trades extra peak memory for fewer reductions and sorts. If `N` is the
number of elements after reducing the ensemble axis and `s` is the bytes per
statistic element, the retained tensor data occupies `3 * N * s` bytes in
quantile mode or `2 * N * s` in std mode, regardless of the alpha/delta grid size.
Here `s=4` for float32, including promoted float16/bfloat16 inputs, and `s=8` for
float64. For example, ten million elements require about 120 MB of quantile-cache
tensor data in float32. These figures exclude input/output tensors, temporary
quantile workspaces, and any autograd intermediates retained
when prediction inputs require gradients. Large spatial grids can therefore
make peak memory significant even with only one alpha cached. Output tensors
still grow with the alpha/delta grid size.

To disable the cache, set `cache_statistics=False`:

```python
calibrator = EnsembleRCPS(
    alphas=[0.1, 0.2],
    deltas=[0.05, 0.1],
    cache_statistics=False,
)
```

This recomputes the mean and widths for every interval evaluation, including
each search step and delta, and releases the statistics after constructing the
interval. It removes the cache's retained tensors at the cost of repeated work;
it does not stream the dataset or remove the memory required for inputs, outputs,
temporary statistics, or quantile computation, so peak memory may still be high.

The calibrator drops its cache references in a `finally` block when the call
returns or raises, so later calls recompute statistics from their current inputs.
Since this temporary state is stored on the instance, calls on the same
calibrator must be serialized. RCPS also retains detached calibration targets
and raw predictions between calls to support `risk_upper_bound`; that storage
outlives the temporary statistics cache.

## Predicted uncertainty half-widths

`ScaledIntervalRCPS` accepts `TensorBNCU`, whose final axis stores the predictive
centre, lower uncertainty half-width, and upper uncertainty half-width. These
half-widths are non-negative distances from the centre, not interval endpoints.
It constructs

$$
T_\lambda(X) = [f(X)-\lambda l(X), f(X)+\lambda u(X)].
$$

```python
import torch

from autouq.calibrators import ScaledIntervalRCPS

pred_cal = torch.stack((centre_cal, lower_width_cal, upper_width_cal), dim=-1)
calibrator = ScaledIntervalRCPS(alphas=0.1, deltas=0.05)
calibrator.calibrate(true_cal, pred_cal)
intervals = calibrator.predict(pred_test)
```

The result has shape
`(batch, *optional_dims, channel, lower_upper, alpha, delta)`. `alphas` are the
risk tolerances and `deltas` are the probabilities with which risk control may
fail over a random calibration-set draw. `predict` evaluates the Cartesian
product in the supplied alpha and delta order.

The guarantee is pointwise for each alpha/delta pair, not simultaneous over the
whole grid. For simultaneous control with total failure probability `delta`,
configure per-pair failure probabilities whose sum is at most `delta`.

Alpha and delta grids are configured up front. `calibrate` fits the complete
Cartesian product eagerly, and `lambda_hat(alpha, delta)` exposes any fitted
scale. This keeps `predict` focused on applying a fully calibrated object and
makes its output axes fixed by the calibrator configuration.

`RCPSCalibrator` is the abstract counterpart of `ConformalCalibrator`: concrete
subclasses define how their prediction tensor produces an interval at a given
`lambda`, while the base owns loss evaluation, risk bounds, and search.

## RCPS and conformal ensemble approaches

The existing [`Ensemble`](ensemble.md) conformal calibrator and `EnsembleRCPS`
accept the same raw ensemble tensor shape but provide different guarantees and
calibration behaviour:

- Conformal `Ensemble(mode="quantile")` builds an interval using a fixed
  `ensemble_alpha`, then applies an additive conformal score correction for each
  alpha requested at prediction time. Its guarantee is marginal coverage.
- `EnsembleRCPS(mode="quantile")` uses each configured risk alpha for both the
  empirical ensemble quantiles and the target expected coverage loss, then fits
  a multiplicative scale for every alpha/delta pair. Its guarantee holds with
  probability at least `1 - delta` over the calibration-set draw.

In both cases, the calibration and test ensembles must be generated by the same
prediction procedure.

### Side-by-side ensemble API

Both approaches accept the same raw ensemble predictions, so they can be
compared without changing the predictive model:

```python
from autouq.calibrators import Ensemble, EnsembleRCPS

alphas = [0.1, 0.2]

cp = Ensemble(mode="quantile", ensemble_alpha=0.1)
rcps = EnsembleRCPS(
    alphas=alphas,
    deltas=[0.05, 0.2],
    mode="quantile",
)

cp.calibrate(true_cal, ensemble_cal)
rcps.calibrate(true_cal, ensemble_cal)

cp_intervals = cp.predict(ensemble_test, alphas=alphas)
rcps_intervals = rcps.predict(ensemble_test)

# CP:   (batch, *optional_dims, channel, 2, alpha)
# RCPS: (batch, *optional_dims, channel, 2, alpha, delta)
```

With the default coverage loss and Hoeffding bound, decreasing the failure
probability delta at a fixed risk level alpha increases the additive correction
in the risk upper-confidence bound. The fitted RCPS scale and interval width
therefore cannot decrease, while coverage loss cannot increase. There is no
required width ordering between CP and RCPS because their calibration objectives
and interval corrections differ.

## Coverage loss and future losses

The default `CoverageLoss` returns one number per calibration example: the
fraction of its temporal, spatial, and channel cells that fall outside their
prediction intervals. Batch items are the independent observations used by the
Hoeffding bound; cells within a batch item are averaged before the empirical
risk is computed.

The calibrators accept a `RiskLoss` and `RiskBound` independently:

```python
from autouq.calibrators import EnsembleRCPS, HoeffdingBound
from autouq.metrics import CoverageLoss

calibrator = EnsembleRCPS(
    alphas=0.1,
    deltas=0.05,
    loss=CoverageLoss(),
    bound=HoeffdingBound(),
)
```

Losses return one finite, non-negative value per batch item and must decrease as
the nested intervals grow. Bounds turn those calibration losses and `delta`
into an upper confidence bound. `CoverageLoss` and `HoeffdingBound` are the only
implementations provided initially. Keeping the contracts separate allows a
future unbounded loss to be paired with an appropriate bound without changing
the prediction subclasses. String or enum lookup for built-in losses and bounds
can be added later without changing these callable contracts.

The base RCPS API therefore accepts any finite, non-negative alpha. The default
ensemble quantile construction additionally requires alpha in `(0, 1)` because
it uses alpha to select percentiles. An unbounded loss with a wider risk scale
can instead use `ScaledIntervalRCPS`, or ensemble `mode="std"`, together with a
suitable bound.

For $n$ calibration examples, its upper confidence bound is

$$
\widehat{R}^{+}(\lambda) = \widehat{R}(\lambda) + \sqrt{\frac{\log(1 / \delta)}{2n}}.
$$

The calibrator expands an initial upper scale until the bound is at most
$\alpha$, then uses binary search to return a feasible scale within
`search_tolerance`. This is valid for the initial coverage-loss/Hoeffding pair:
the intervals are nested, coverage loss is non-increasing as they grow, and the
Hoeffding upper bound preserves that ordering. Custom losses and bounds used
with this search must preserve the same monotonicity. The calibrator raises an
error when expanding the prediction intervals cannot satisfy the configured
loss and bound at the requested risk level.
