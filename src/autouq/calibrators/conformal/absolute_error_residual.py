from collections.abc import Sequence

import torch

from autouq.calibrators.conformal.conformal import ConformalCalibrator
from autouq.types import Tensor, TensorBNC, TensorBNIA


class AbsoluteErrorResidual(ConformalCalibrator[TensorBNC]):
    """Absolute Error Residual base class."""

    def _score(
        self,
        true: TensorBNC,
        pred: TensorBNC,
    ) -> Tensor:
        if true.shape != pred.shape:
            msg = (
                "true and pred must have the same shape; "
                f"got {tuple(true.shape)} and {tuple(pred.shape)}."
            )
            raise ValueError(msg)
        return torch.abs(true - pred)

    def _predict(self, pred: TensorBNC, alphas: Sequence[float]) -> TensorBNIA:
        scores = self._calibration_scores()
        if pred.shape[1:] != scores.shape[1:]:
            msg = (
                "pred must match the calibrated trailing shape; "
                f"got {tuple(pred.shape[1:])} and expected "
                f"{tuple(scores.shape[1:])}."
            )
            raise ValueError(msg)

        intervals = []
        for alpha in alphas:
            score_quantile = self.score_quantile(alpha).to(device=pred.device)
            intervals.append(
                torch.stack((pred - score_quantile, pred + score_quantile), dim=-1)
            )
        return torch.stack(intervals, dim=-1)
