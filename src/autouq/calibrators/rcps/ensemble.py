import math
from collections.abc import Sequence
from enum import StrEnum

import torch

from autouq.calibrators.rcps.bounds import RiskBound
from autouq.calibrators.rcps.rcps import RCPSCalibrator
from autouq.metrics import RiskLoss
from autouq.types import TensorBNC, TensorBNCM, TensorBNI, TensorBNIAD


class _EnsembleMode(StrEnum):
    QUANTILE = "quantile"
    STD = "std"


class EnsembleRCPS(RCPSCalibrator[TensorBNCM]):
    """RCPS for raw ensemble forecasts.

    The final prediction axis indexes ensemble members. In the default
    ``"quantile"`` mode, empirical alpha/2 and 1-alpha/2 quantiles define
    asymmetric widths around the ensemble mean. ``"std"`` mode instead uses
    the ensemble population standard deviation as a symmetric width.
    Float16 and bfloat16 predictions are promoted to float32 for both ensemble
    statistics and output intervals, preserving the positive width floor.

    Within each ``calibrate`` or ``predict`` call, a temporary cache stores the
    ensemble mean and clamped lower/upper half-widths, each shaped like ``pred``
    without its member axis. These depend on the input and, in quantile mode,
    alpha, but not on lambda or delta. Reusing them avoids repeating ensemble
    reductions and quantile sorts during the scale search and across deltas.
    Quantile mode retains three tensors for only the current alpha; std mode
    shares one mean and one width tensor across all pairs. Set
    ``cache_statistics=False`` to recompute them for every interval evaluation.
    Cache references are cleared on return or error. Calls on the same instance
    must be serialized.

    Args:
        alphas: Target risk levels. A scale is fitted for each alpha/delta pair.
        deltas: Risk-control failure probabilities.
        mode: Ensemble interval construction mode. Supported values are
            ``"quantile"`` and ``"std"``.
        min_scale: Minimum lower/upper width used for numerical stability.
        cache_statistics: Reuse the current alpha's ensemble statistics within
            each call. Defaults to True. False trades repeated computation for
            releasing statistics between interval evaluations.
        loss: Per-example risk loss. Defaults to coverage loss.
        bound: Risk upper confidence bound. Defaults to the Hoeffding bound.
        search_tolerance: Absolute binary-search tolerance for lambda.
        max_search_steps: Maximum expansion and binary-search iterations.
        temporal_dim: Optional temporal dimension index.
        spatial_dims: Optional spatial dimension indices.
    """

    def __init__(
        self,
        alphas: float | Sequence[float],
        deltas: float | Sequence[float],
        loss: RiskLoss | None = None,
        bound: RiskBound | None = None,
        *,
        mode: str = "quantile",
        min_scale: float = 1e-8,
        cache_statistics: bool = True,
        search_tolerance: float = 1e-4,
        max_search_steps: int = 64,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        try:
            self.mode = _EnsembleMode(mode)
        except ValueError as exc:
            msg = f"mode must be 'quantile' or 'std', got {mode!r}."
            raise ValueError(msg) from exc
        if not math.isfinite(min_scale) or min_scale <= 0:
            msg = "min_scale must be finite and greater than zero."
            raise ValueError(msg)
        self.min_scale = min_scale
        self.cache_statistics = cache_statistics
        # When enabled, keep at most one (centre, lower_width, upper_width)
        # tuple during calibrate/predict, keyed by alpha (or 0.0 in std mode).
        self._statistics_cache: (
            dict[float, tuple[TensorBNC, TensorBNC, TensorBNC]] | None
        ) = None
        super().__init__(
            alphas=alphas,
            deltas=deltas,
            loss=loss,
            bound=bound,
            search_tolerance=search_tolerance,
            max_search_steps=max_search_steps,
            temporal_dim=temporal_dim,
            spatial_dims=spatial_dims,
        )
        if self.mode is _EnsembleMode.QUANTILE and any(
            not 0 < alpha < 1 for alpha in self.alphas
        ):
            msg = (
                "quantile mode requires every alpha to be in the open interval (0, 1)."
            )
            raise ValueError(msg)

    def calibrate(self, true: TensorBNC, pred: TensorBNCM) -> None:
        """Fit scales, reusing ensemble statistics throughout the search."""
        self._statistics_cache = {} if self.cache_statistics else None
        try:
            super().calibrate(true, pred)
        finally:
            self._statistics_cache = None

    def predict(self, pred: TensorBNCM) -> TensorBNIAD:
        """Apply fitted scales, reusing ensemble statistics across deltas."""
        self._statistics_cache = {} if self.cache_statistics else None
        try:
            return super().predict(pred)
        finally:
            self._statistics_cache = None

    @staticmethod
    def _validate_ensemble(pred: TensorBNCM) -> None:
        if pred.ndim < 3 or pred.shape[-1] < 2:
            msg = (
                "pred must have a final ensemble axis with at least two members; "
                f"got shape {tuple(pred.shape)}."
            )
            raise ValueError(msg)
        if pred.shape[0] == 0:
            msg = "pred must contain at least one batch item."
            raise ValueError(msg)

    def _validate_calibration_prediction(
        self, true: TensorBNC, pred: TensorBNCM
    ) -> None:
        self._validate_ensemble(pred)
        if pred.shape[:-1] != true.shape:
            msg = (
                "true must match pred without its ensemble axis; got "
                f"{tuple(true.shape)} and expected {tuple(pred.shape[:-1])}."
            )
            raise ValueError(msg)

    def _validate_test_prediction(
        self, pred: TensorBNCM, calibration_pred: TensorBNCM
    ) -> None:
        self._validate_ensemble(pred)
        if pred.shape[1:] != calibration_pred.shape[1:]:
            msg = (
                "pred must match the calibrated trailing shape, including its "
                f"ensemble size; got {tuple(pred.shape[1:])} and expected "
                f"{tuple(calibration_pred.shape[1:])}."
            )
            raise ValueError(msg)

    def _ensemble_statistics(
        self, pred: TensorBNCM, alpha: float
    ) -> tuple[TensorBNC, TensorBNC, TensorBNC]:
        """Return the mean and clamped half-widths with the member axis removed."""
        # Lambda changes only the final interval arithmetic; delta changes only
        # the risk bound. Neither requires recomputing these ensemble statistics.
        key = alpha if self.mode is _EnsembleMode.QUANTILE else 0.0
        if self._statistics_cache is not None:
            if key in self._statistics_cache:
                return self._statistics_cache[key]
            # Both base-class loops finish all deltas for one alpha before
            # advancing. Release that alpha's tensors before computing the next.
            self._statistics_cache.clear()

        # The default floor underflows in float16; quantile also requires at
        # least float32. Keep the interval arithmetic in the promoted dtype.
        if pred.dtype in (torch.float16, torch.bfloat16):
            pred = pred.float()
        centre = pred.mean(dim=-1)
        if self.mode is _EnsembleMode.QUANTILE:
            lower = torch.quantile(pred, alpha / 2, dim=-1)
            upper = torch.quantile(pred, 1 - alpha / 2, dim=-1)
            lower_width = (centre - lower).clamp_min(self.min_scale)
            upper_width = (upper - centre).clamp_min(self.min_scale)
        else:
            scale = pred.std(dim=-1, correction=0).clamp_min(self.min_scale)
            lower_width = scale
            upper_width = scale
        statistics = centre, lower_width, upper_width
        if self._statistics_cache is not None:
            self._statistics_cache[key] = statistics
        return statistics

    def _prediction_set(
        self, pred: TensorBNCM, lambda_: float, alpha: float
    ) -> TensorBNI:
        centre, lower_width, upper_width = self._ensemble_statistics(pred, alpha)
        return torch.stack(
            (
                centre - lambda_ * lower_width,
                centre + lambda_ * upper_width,
            ),
            dim=-1,
        )
