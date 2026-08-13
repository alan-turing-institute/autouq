# Conformal Calibrators

Conformal calibrators turn model predictions and held-out calibration targets
into prediction intervals. They do not fit the underlying predictive model; they
only calibrate tensors that have already been produced.

## Base API

`Calibrator[PredT]` is the root abstraction. `PredT` lets subclasses specialize
the prediction tensor contract while keeping the public method names stable.

The target tensor is channels-last and uses the `TensorBNC` layout:

```text
(batch, *optional_dims, channel)
```

The first axis is the calibration or prediction batch. The optional dimensions
can represent temporal axes, spatial axes, or neither. `temporal_dim` and
`spatial_dims` describe those optional axes for callers and subclasses; they do
not currently rearrange tensors.

`ConformalCalibrator[PredT, ScoreT, ScoreQuantileT]` adds the split-conformal
workflow:

```python
calibrator = SomeConformalCalibrator(...)
calibrator.calibrate(true_cal, pred_cal)
intervals = calibrator.predict(pred_test, alphas=[0.1, 0.2])
```

`calibrate` computes and caches calibration scores. `predict` normalizes and
validates `alphas`, then delegates interval construction to the subclass.
Subclasses use the generic type parameters to document their tensor contract:

- `PredT`: Prediction tensor passed to `calibrate` and `predict`.
- `ScoreT`: Cached calibration-score tensor with calibration examples on
  dimension 0.
- `ScoreQuantileT`: Score-quantile tensor after the calibration dimension has
  been removed.

`score_quantile(alpha)` treats dimension 0 as the calibration-example axis. If
scores have shape `(n_calibration, *score_dims)`, the quantile is computed
across the `n_calibration` examples for each fixed trailing score cell. The
result keeps the trailing score structure:

```text
(*score_dims)
```

Prediction intervals use the `TensorBNIA` layout:

```text
(batch, *optional_dims, channel, 2, alphas)
```

The interval axis has size 2 and stores lower then upper bounds.

## Streaming Calibration

`calibrate`/`cache_scores` overwrite the cached scores in one shot and need
the full calibration tensor in memory at once. When calibration examples
must instead be accumulated one at a time or in small batches (e.g. one
example is already too large to hold many of at once), use the streaming
counterpart:

```python
calibrator.reset()
for true_chunk, pred_chunk in calibration_chunks:
    calibrator.update(true_chunk, pred_chunk)
intervals = calibrator.predict(pred_test, alphas=[0.1, 0.2])
```

Chunks are concatenated lazily - only once, the first time `score_quantile`
or `predict` actually needs the full calibration score tensor - so many
small chunks can be streamed in without ever materializing the full bank
until it's required. `reset` clears any cached or pending scores;
`calibrate`/`cache_scores` still work as before and discard any pending
streamed chunks when called.

To read out the full calibration-scores tensor directly - e.g. to combine
this calibrator's locally-streamed scores with scores accumulated elsewhere,
before deciding on a final `alpha` - call `materialize_scores()` rather than
inspecting `scores` directly: `scores` stays `None` until something forces
materialization, while `materialize_scores()` triggers it (concatenating any
pending chunks, at most once) and returns the result, or `None` if nothing
has been accumulated yet.

Some calibrators (e.g. [`Ensemble`](ensemble.md)) add a further, subclass-specific
streaming layer for computing one calibration example's score itself, feeding
`accumulate_score` in turn - see that calibrator's docs for details.

## Concrete Calibrators

- [`AbsoluteErrorResidual`](absolute_error_residual.md) calibrates point
  predictions with absolute residual scores and symmetric intervals.
- [`ConformalizedQuantileRegression`](conformalized_quantile_regression.md)
  calibrates lower and upper quantile predictions.
- [`Ensemble`](ensemble.md) calibrates ensemble predictions using empirical
  ensemble intervals or normalized residuals.

## Expectations

- Call `calibrate` before `predict` or `score_quantile`.
- Pass `alpha` values strictly between 0 and 1.
- Put calibration examples on dimension 0.
- Keep prediction tensors compatible with the subclass contract.
