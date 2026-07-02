# Conformalized Quantile Regression

`ConformalizedQuantileRegression` calibrates lower and upper quantile
pred tensors. The configured `quantile_level_pairs` are the lower/upper quantile
levels produced by the quantile model. The calibrator derives prediction
miscoverage levels from those pairs.

## Workflow

```python
from autouq.calibrators import ConformalizedQuantileRegression

calibrator = ConformalizedQuantileRegression(
    quantile_level_pairs=[(0.05, 0.95)],
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

For one quantile pair, `pred` uses `TensorBNCQ`:

```text
(batch, *optional_dims, channel, 2)
```

For multiple quantile pairs, `pred` uses `TensorBNCQK`:

```text
(batch, *optional_dims, channel, 2, quantile_pairs)
```

The lower/upper quantile axis has size 2 and stores lower then upper quantile
pred tensors. In the multi-pair case, the final `quantile_pairs` axis indexes
the configured quantile pairs, so it has size `len(quantile_level_pairs)`.

For example, configured quantile-level pairs `[(0.05, 0.95), (0.1, 0.9)]` are
selected at prediction time with miscoverage levels `0.1` and `0.2`,
respectively. If a quantile model returns a flat raw-quantile axis, arrange
those pred tensors into lower/upper pairs before passing them to this
calibrator.
The `alphas` passed to `predict` must match the alphas derived from these
quantile pairs.

In base-class terms, this calibrator specializes:

```text
ConformalCalibrator[
    TensorBNCQ | TensorBNCQK,
    TensorBNC | TensorBNCK,
    TensorNC | TensorNCK,
]
```

For one quantile pair, cached scores use `TensorBNC`. For multiple quantile
pairs, cached scores use `TensorBNCK` so each configured quantile pair has its
own conformal correction. The public `score_quantile(alpha)` method selects the
quantile pair matching the requested prediction miscoverage level and returns a
`TensorNC` threshold.

Returned intervals use `TensorBNIA`:

```text
(batch, *optional_dims, channel, 2, alphas)
```

The returned final alpha axis follows the prediction miscoverage levels passed
to `predict`.

## References

- Romano, Patterson, and Candes (2019), [Conformalized Quantile
  Regression](https://arxiv.org/abs/1905.03222).
