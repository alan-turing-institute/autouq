import math
from collections.abc import Sequence
from enum import StrEnum

import torch

from autouq.calibrators.rcps.bounds import RiskBound
from autouq.calibrators.rcps.rcps import RCPSCalibrator
from autouq.metrics import RiskLoss
from autouq.types import TensorBNC, TensorBNCM, TensorBNI


class _EnsembleMode(StrEnum):
    QUANTILE = "quantile"
    STD = "std"


class EnsembleRCPS(RCPSCalibrator[TensorBNCM]):
    """RCPS for raw ensemble forecasts.

    The final prediction axis indexes ensemble members. In the default
    ``"quantile"`` mode, empirical alpha/2 and 1-alpha/2 quantiles define
    asymmetric widths around the ensemble mean. ``"std"`` mode instead uses
    the ensemble population standard deviation as a symmetric width.

    Args:
        alphas: Target risk levels. A scale is fitted for each alpha/delta pair.
        deltas: Risk-control failure probabilities.
        mode: Ensemble interval construction mode. Supported values are
            ``"quantile"`` and ``"std"``.
        min_scale: Minimum lower/upper width used for numerical stability.
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

    def _prediction_set(
        self, pred: TensorBNCM, lambda_: float, alpha: float
    ) -> TensorBNI:
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
        return torch.stack(
            (
                centre - lambda_ * lower_width,
                centre + lambda_ * upper_width,
            ),
            dim=-1,
        )
