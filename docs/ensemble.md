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

For an alternative that controls expected coverage loss with a user-specified
failure probability over the calibration-set draw, see
[`EnsembleRCPS`](rcps.md#rcps-and-conformal-ensemble-approaches).

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
