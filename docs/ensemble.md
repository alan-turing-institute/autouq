# Ensemble Conformal Calibrator

`Ensemble` calibrates ensemble predictions where the final prediction dimension
indexes ensemble members.

## Workflow

```python
from autouq.calibrators import Ensemble

calibrator = Ensemble(
    temporal_dim=1,
    spatial_dims=(2, 3),
    mode="quantile",
    ensemble_alpha=0.1,
)
calibrator.calibrate(true_cal, pred_cal)
intervals = calibrator.predict(pred_test, alphas=[0.1, 0.2])
```

## Modes

`mode="quantile"` builds a lower and upper empirical quantile interval from the
ensemble, then conformalizes that interval with additive score thresholds.

`mode="std"` uses the ensemble mean as the center and the ensemble standard
deviation as the scale. Calibration scores are normalized residuals, and
prediction intervals scale the conformal threshold by the prediction-time
ensemble standard deviation.

## Tensor Contract

`true` uses the channels-last `TensorBNC` layout:

```text
(batch, *optional_dims, channel)
```

`pred` uses `TensorBNCM`:

```text
(batch, *optional_dims, channel, ensemble)
```

The final ensemble axis must contain at least two members.

In base-class terms, this calibrator specializes:

```text
ConformalCalibrator[TensorBNCM, TensorBNC, TensorNC]
```

Cached calibration scores use `TensorBNC`, and `score_quantile(alpha)` returns
a `TensorNC` threshold.

Returned intervals use `TensorBNIA`:

```text
(batch, *optional_dims, channel, 2, alphas)
```

## Streaming Members

The whole-tensor `calibrate`/`predict` API needs the full `(*, channel,
ensemble)` tensor for an example in memory at once. When that tensor is too
large to materialize - e.g. many members over a large spatial grid - stream
one member at a time instead:

```python
# Outer loop: one iteration per calibration example (e.g. one forecast
# initialization time). Each example's raw members are only ever held one
# at a time, and are discarded once that example's score has been folded
# into the calibration bank via accumulate_score.
for true_for_this_example, members in calibration_examples:
    calibrator.reset_online()
    for member in members:                  # one model forward pass each
        calibrator.update_online(member, true=true_for_this_example)
    score = calibrator.finalize_online(chunk_size=2048)   # chunk_size optional
    calibrator.accumulate_score(score)       # feed the base calibration layer

intervals = calibrator.predict(pred_test, alphas=[0.1, 0.2])
```

`update_online` accepts one member's prediction, without an ensemble
dimension (i.e. `TensorBNC`, not `TensorBNCM`). `true` only needs to be
passed once (e.g. alongside the first member) - it's identical across
members for a given example.

`finalize_online` returns the score for that one example (the same value
`calibrate` would compute from the whole tensor), ready to hand to
`accumulate_score`. For `mode="std"`, this is fully incremental - members are
never retained, only a running mean and sum-of-squared-deviations (Welford's
online algorithm) - so the ensemble axis is never materialized regardless of
`chunk_size`. For `mode="quantile"`, `torch.quantile` still needs every
member, so members are retained; passing `chunk_size` tiles that quantile
computation over `chunk_dim` (default: the axis immediately before
`channel`, typically a spatial axis) so the full retained-member stack is
combined one chunk at a time instead of all at once.

The same member-streaming machinery has a prediction-time counterpart,
`finalize_online_interval(alphas, chunk_size=..., chunk_dim=...)`, which
skips the score computation (no `true` needed) and returns calibrated
intervals directly for the streamed example - the streaming equivalent of
`predict` for one example's members.

## Validity Notes

This class applies split conformal calibration to ensemble-derived intervals or
scales. The ensemble is the base prediction mechanism; validity still relies on
the calibration and prediction scores being exchangeable under the same
prediction-generation procedure.

For stochastic ensembles, generate calibration and prediction ensemble tensors
with the same fitted model, ensemble size, and sampling protocol. The ensemble
randomness can be treated as part of the prediction procedure, but it should be
applied consistently across calibration and prediction examples.

This class does not implement EnbPI. EnbPI is a related ensemble conformal method
for time series that uses bootstrap ensemble predictors and avoids the standard
exchangeability requirement.

## References

- Lei, G'Sell, Rinaldo, Tibshirani, and Wasserman (2018),
  [Distribution-Free Predictive Inference for Regression](https://arxiv.org/abs/1604.04173).
  This is the split-conformal regression reference for wrapping arbitrary base
  predictors.
- Romano, Patterson, and Candes (2019), [Conformalized Quantile
  Regression](https://arxiv.org/abs/1905.03222). This is the interval-score
  conformalization reference relevant to `mode="quantile"`.
- Xu and Xie (2023), [Conformal Prediction for Time
  Series](https://arxiv.org/abs/2010.09107). This introduces EnbPI, a related
  ensemble conformal method for time-series prediction.
- Jensen, Bianchi, and Anfinsen (2022), [Ensemble Conformalized Quantile
  Regression for Probabilistic Time Series
  Forecasting](https://arxiv.org/abs/2202.08756). This is related ensemble+CQR
  work for probabilistic time-series forecasting.
