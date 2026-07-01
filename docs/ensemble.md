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

The final ensemble axis must contain at least two members. Returned intervals use
`TensorBNIA`:

```text
(batch, *optional_dims, channel, 2, alphas)
```
