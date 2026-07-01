from collections.abc import Sequence

import torch

from autouq.calibrators.conformal.conformal import ConformalCalibrator
from autouq.types import Tensor, TensorBNC, TensorBNIA


class AbsoluteErrorResidual(ConformalCalibrator[TensorBNC]):
    r"""Split-conformal calibrator using absolute residual scores.

    Calibration scores are computed cell-wise as

    .. math::

        s_i = |y_i - \hat{y}_i|.

    Prediction intervals are symmetric around the point prediction:

    .. math::

        [\hat{y} - \hat{q}_{1-\alpha},
        \hat{y} + \hat{q}_{1-\alpha}].

    Args:
        temporal_dim: Optional index of the temporal dimension in tensors passed
            to the calibrator.
        spatial_dims: Optional indices of spatial dimensions in tensors passed
            to the calibrator.
    """

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
