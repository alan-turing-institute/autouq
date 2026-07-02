from collections.abc import Sequence
from enum import StrEnum

import torch

from autouq.calibrators.conformal.conformal import ConformalCalibrator
from autouq.calibrators.grouping import validate_alpha
from autouq.types import TensorBNC, TensorBNCM, TensorBNIA, TensorNC


class _EnsembleMode(StrEnum):
    QUANTILE = "quantile"
    STD = "std"


class Ensemble(ConformalCalibrator[TensorBNCM, TensorBNC, TensorNC]):
    """Conformal calibrator for ensemble forecasts.

    The final ``pred`` dimension indexes ensemble members. ``mode="quantile"``
    conformalizes an empirical ensemble quantile interval, while ``mode="std"``
    calibrates normalized residuals around the ensemble mean.

    Args:
        temporal_dim: Optional index of the temporal dimension in tensors passed
            to the calibrator.
        spatial_dims: Optional indices of spatial dimensions in tensors passed
            to the calibrator.
        mode: Ensemble interval construction mode. Supported values are
            ``"quantile"`` and ``"std"``.
        ensemble_alpha: Miscoverage level used for the empirical ensemble
            quantile interval in ``"quantile"`` mode.
        min_scale: Minimum scale used to stabilize ``"std"`` mode.
    """

    def __init__(
        self,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
        mode: str = "quantile",
        ensemble_alpha: float = 0.1,
        min_scale: float = 1e-8,
    ):
        super().__init__(temporal_dim=temporal_dim, spatial_dims=spatial_dims)
        try:
            self.mode = _EnsembleMode(mode)
        except ValueError as exc:
            msg = f"mode must be 'quantile' or 'std', got {mode!r}."
            raise ValueError(msg) from exc
        validate_alpha(ensemble_alpha, name="ensemble_alpha")
        if min_scale <= 0:
            msg = f"min_scale must be positive, got {min_scale}."
            raise ValueError(msg)

        self.ensemble_alpha = ensemble_alpha
        self.min_scale = min_scale

    def _score(self, true: TensorBNC, pred: TensorBNCM) -> TensorBNC:
        ensemble = self._validate_ensemble_predictions(pred)
        if true.shape != ensemble.shape[:-1]:
            msg = (
                "true must match the ensemble pred shape without the "
                f"ensemble dimension; got {tuple(true.shape)} and expected "
                f"{tuple(ensemble.shape[:-1])}."
            )
            raise ValueError(msg)

        if self.mode is _EnsembleMode.QUANTILE:
            lower, upper = self._quantile_interval(ensemble)
            return torch.maximum(lower - true, true - upper)

        center, scale = self._mean_and_scale(ensemble)
        return torch.abs(true - center) / scale

    def _predict(self, pred: TensorBNCM, alphas: Sequence[float]) -> TensorBNIA:
        scores = self._calibration_scores()
        ensemble = self._validate_ensemble_predictions(pred)
        if ensemble.shape[1:-1] != scores.shape[1:]:
            msg = (
                "pred must match the calibrated trailing shape; "
                f"got {tuple(ensemble.shape[1:-1])} and expected "
                f"{tuple(scores.shape[1:])}."
            )
            raise ValueError(msg)

        intervals = []
        if self.mode is _EnsembleMode.QUANTILE:
            lower, upper = self._quantile_interval(ensemble)
            for alpha in alphas:
                score_quantile = self.score_quantile(alpha).to(device=pred.device)
                intervals.append(
                    torch.stack(
                        (lower - score_quantile, upper + score_quantile), dim=-1
                    )
                )
            return torch.stack(intervals, dim=-1)

        center, scale = self._mean_and_scale(ensemble)
        for alpha in alphas:
            score_quantile = self.score_quantile(alpha).to(device=pred.device)
            intervals.append(
                torch.stack(
                    (
                        center - score_quantile * scale,
                        center + score_quantile * scale,
                    ),
                    dim=-1,
                )
            )
        return torch.stack(intervals, dim=-1)

    def _validate_ensemble_predictions(self, pred: TensorBNCM) -> TensorBNCM:
        if pred.ndim < 2 or pred.shape[-1] < 2:
            msg = (
                "Ensemble predictions must include a final ensemble dimension "
                f"with at least 2 members; got shape {tuple(pred.shape)}."
            )
            raise ValueError(msg)
        return pred

    def _quantile_interval(self, ensemble: TensorBNCM) -> tuple[TensorBNC, TensorBNC]:
        lower_q = self.ensemble_alpha / 2
        upper_q = 1 - lower_q
        lower = torch.quantile(ensemble, lower_q, dim=-1)
        upper = torch.quantile(ensemble, upper_q, dim=-1)
        return lower, upper

    def _mean_and_scale(self, ensemble: TensorBNCM) -> tuple[TensorBNC, TensorBNC]:
        center = ensemble.mean(dim=-1)
        scale = ensemble.std(dim=-1, correction=0).clamp_min(self.min_scale)
        return center, scale
