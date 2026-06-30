from collections.abc import Callable, Sequence

import torch

from autouq.calibrators.conformal.conformal import ConformalCalibrator
from autouq.types import Tensor, TensorBTSIA


class AbsoluteErrorResidual(ConformalCalibrator):
    """Absolute Error Residual base class."""

    def _score_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        def absolute_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
            if y_true.shape != y_pred.shape:
                msg = (
                    "y_true and y_pred must have the same shape; "
                    f"got {tuple(y_true.shape)} and {tuple(y_pred.shape)}."
                )
                raise ValueError(msg)
            return torch.abs(y_true - y_pred)

        return absolute_error

    def _predict(self, y_pred: Tensor, alphas: Sequence[float]) -> TensorBTSIA:
        scores = self._calibration_scores()
        if y_pred.shape[1:] != scores.shape[1:]:
            msg = (
                "y_pred must match the calibrated trailing shape; "
                f"got {tuple(y_pred.shape[1:])} and expected "
                f"{tuple(scores.shape[1:])}."
            )
            raise ValueError(msg)

        intervals = []
        for alpha in alphas:
            q_hat = self.q_hat(alpha).to(device=y_pred.device)
            intervals.append(torch.stack((y_pred - q_hat, y_pred + q_hat), dim=-1))
        return torch.stack(intervals, dim=-1)
