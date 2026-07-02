# Absolute Error Residual Calibrator

`AbsoluteErrorResidual` is the standard split-conformal calibrator for point
predictions. It assumes the model prediction is a center estimate and calibrates
symmetric intervals around that prediction.

## Workflow

```python
from autouq.calibrators import AbsoluteErrorResidual

calibrator = AbsoluteErrorResidual(temporal_dim=1, spatial_dims=(2, 3))
calibrator.calibrate(true_cal, pred_cal)
intervals = calibrator.predict(pred_test, alphas=0.1)
```

During calibration, the score is the absolute residual:

```text
score = abs(true - pred)
```

At prediction time, the calibrator computes the finite-sample conformal score
threshold for each requested `alpha` and returns:

```text
[pred - score_quantile(alpha), pred + score_quantile(alpha)]
```

For `n_calibration` calibration examples, `score_quantile(alpha)` uses the
one-indexed order statistic:

```text
k = ceil((n_calibration + 1) * (1 - alpha))
```

The threshold is the `k`th smallest calibration score for each fixed trailing
cell.

## Tensor Contract

`AbsoluteErrorResidual` specializes the conformal base as:

```text
ConformalCalibrator[TensorBNC, TensorBNC, TensorNC]
```

The first type is the prediction tensor, the second is the cached score tensor,
and the third is the score-quantile tensor.

`true` and `pred` use the channels-last `TensorBNC` layout:

```text
(batch, *optional_dims, channel)
```

The calibration batch is dimension 0. Any optional temporal or spatial structure
is preserved cell-by-cell when computing score quantiles. Prediction tensors must
match the calibrated trailing shape:

```text
pred.shape[1:] == scores.shape[1:]
```

The score quantile returned by `score_quantile(alpha)` uses `TensorNC`:

```text
(*optional_dims, channel)
```

The returned intervals use `TensorBNIA`:

```text
(batch, *optional_dims, channel, 2, alphas)
```

The interval axis stores lower then upper bounds.

## References

- Lei, G'Sell, Rinaldo, Tibshirani, and Wasserman (2018), [Distribution-Free
  Predictive Inference for Regression](https://arxiv.org/abs/1604.04173).
  This is the main split-conformal regression reference for this calibrator.
