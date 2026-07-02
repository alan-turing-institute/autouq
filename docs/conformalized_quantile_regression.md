# Conformalized Quantile Regression

`ConformalizedQuantileRegression` calibrates lower and upper quantile
predictions. The configured `alphas` describe which quantile pairs are supplied
in `pred`.

## Workflow

```python
from autouq.calibrators import ConformalizedQuantileRegression

calibrator = ConformalizedQuantileRegression(
    alphas=0.1,
    temporal_dim=1,
    spatial_dims=(2, 3),
)
calibrator.calibrate(true_cal, pred_cal)
intervals = calibrator.predict(pred_test, alphas=0.1)
```

During calibration, the score is:

```text
score = max(lower - true, true - upper)
```

At prediction time, the calibrator returns:

```text
[lower - score_quantile(alpha), upper + score_quantile(alpha)]
```

## Tensor Contract

`true` uses the channels-last `TensorBNC` layout:

```text
(batch, *optional_dims, channel)
```

For one configured alpha, `pred` uses `TensorBNCQ`:

```text
(batch, *optional_dims, channel, 2)
```

For multiple configured alphas, `pred` uses `TensorBNCQA`:

```text
(batch, *optional_dims, channel, 2, alphas)
```

The quantile-pair axis has size 2 and stores lower then upper quantiles. In the
multi-alpha case, the final alpha axis must match the calibrator's configured
`alphas`.

In base-class terms, this calibrator specializes:

```text
ConformalCalibrator[
    TensorBNCQ | TensorBNCQA,
    TensorBNC | TensorBNCA,
    TensorNC | TensorNCA,
]
```

For one configured alpha, cached scores use `TensorBNC`. For multiple configured
alphas, cached scores use `TensorBNCA` so each configured alpha has its own
conformal correction. The public `score_quantile(alpha)` method selects the
requested alpha and returns a `TensorNC` threshold.

Returned intervals use `TensorBNIA`:

```text
(batch, *optional_dims, channel, 2, alphas)
```

## References

- Romano, Patterson, and Candes (2019), [Conformalized Quantile
  Regression](https://arxiv.org/abs/1905.03222).
