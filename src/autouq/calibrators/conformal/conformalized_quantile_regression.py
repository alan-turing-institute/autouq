import math
from collections.abc import Sequence

import torch

from autouq.calibrators.conformal.conformal import ConformalCalibrator
from autouq.types import (
    Tensor,
    TensorBNC,
    TensorBNCA,
    TensorBNCQ,
    TensorBNCQA,
    TensorBNIA,
    TensorNC,
    TensorNCA,
)


class ConformalizedQuantileRegression(
    ConformalCalibrator[
        TensorBNCQ | TensorBNCQA,
        TensorBNC | TensorBNCA,
        TensorNC | TensorNCA,
    ]
):
    r"""Conformalized quantile regression calibrator.

    Calibration scores are computed cell-wise from lower and upper quantile
    predictions:

    .. math::

        s_i = \max(\hat{q}_{\mathrm{lo}}(x_i) - y_i,
        y_i - \hat{q}_{\mathrm{hi}}(x_i)).

    Prediction intervals expand the quantile pair by the conformal score
    threshold:

    .. math::

        [\hat{q}_{\mathrm{lo}}(x) - \hat{s}_{1-\alpha},
        \hat{q}_{\mathrm{hi}}(x) + \hat{s}_{1-\alpha}].

    Args:
        alphas: Miscoverage levels corresponding to the quantile pairs in
            ``pred``.
        temporal_dim: Optional index of the temporal dimension in tensors passed
            to the calibrator.
        spatial_dims: Optional indices of spatial dimensions in tensors passed
            to the calibrator.
    """

    def __init__(
        self,
        alphas: float | Sequence[float],
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        super().__init__(temporal_dim=temporal_dim, spatial_dims=spatial_dims)
        self.alphas = self._normalize_alphas(alphas)

    def _score(
        self,
        true: TensorBNC,
        pred: TensorBNCQ | TensorBNCQA,
    ) -> TensorBNC | TensorBNCA:
        lower, upper = self._split_quantile_predictions(pred)
        target_shape = (
            lower.shape[:-1] if self._has_alpha_dimension(lower) else lower.shape
        )
        if true.shape != target_shape:
            msg = (
                "true must match the quantile pred shape; "
                f"got {tuple(true.shape)} and expected {tuple(target_shape)}."
            )
            raise ValueError(msg)
        self._validate_quantile_order(lower, upper)
        if self._has_alpha_dimension(lower):
            true = true.unsqueeze(-1)
        return torch.maximum(lower - true, true - upper)

    def score_quantile(self, alpha: float) -> TensorNC:
        alpha_idx = self._alpha_index(alpha)
        score_quantile = super().score_quantile(alpha)
        if len(self.alphas) == 1:
            return score_quantile
        return score_quantile[..., alpha_idx]

    def _predict(
        self,
        pred: TensorBNCQ | TensorBNCQA,
        alphas: Sequence[float],
    ) -> TensorBNIA:
        scores = self._calibration_scores()
        lower, upper = self._split_quantile_predictions(pred)
        if lower.shape[1:] != scores.shape[1:]:
            msg = (
                "pred must match the calibrated trailing shape; "
                f"got {tuple(lower.shape[1:])} and expected "
                f"{tuple(scores.shape[1:])}."
            )
            raise ValueError(msg)
        self._validate_quantile_order(lower, upper)

        intervals = []
        for alpha in alphas:
            alpha_idx = self._alpha_index(alpha)
            lower_alpha = self._select_alpha(lower, alpha_idx)
            upper_alpha = self._select_alpha(upper, alpha_idx)
            score_quantile = self.score_quantile(alpha).to(device=pred.device)
            intervals.append(
                torch.stack(
                    (lower_alpha - score_quantile, upper_alpha + score_quantile),
                    dim=-1,
                )
            )
        return torch.stack(intervals, dim=-1)

    def _split_quantile_predictions(self, pred: Tensor) -> tuple[Tensor, Tensor]:
        if len(self.alphas) == 1:
            if pred.ndim < 2 or pred.shape[-1] != 2:
                msg = (
                    "CQR predictions for one alpha must include a final "
                    f"lower/upper quantile dimension of size 2; got shape "
                    f"{tuple(pred.shape)}."
                )
                raise ValueError(msg)
            return pred[..., 0], pred[..., 1]

        if pred.ndim < 3 or pred.shape[-2] != 2:
            msg = (
                "CQR predictions for multiple alphas must include a lower/upper "
                f"quantile dimension of size 2 before the alpha dimension; "
                f"got shape {tuple(pred.shape)}."
            )
            raise ValueError(msg)
        if pred.shape[-1] != len(self.alphas):
            msg = (
                "CQR predictions must include one quantile pair per configured "
                f"alpha; got {pred.shape[-1]} alpha predictions and expected "
                f"{len(self.alphas)}."
            )
            raise ValueError(msg)
        return pred[..., 0, :], pred[..., 1, :]

    def _validate_quantile_order(self, lower: Tensor, upper: Tensor) -> None:
        if torch.any(lower > upper):
            msg = "Lower quantile predictions must not exceed upper predictions."
            raise ValueError(msg)

    def _alpha_index(self, alpha: float) -> int:
        self._validate_alpha(alpha)
        for idx, configured_alpha in enumerate(self.alphas):
            if math.isclose(alpha, configured_alpha, rel_tol=1e-12, abs_tol=1e-12):
                return idx
        msg = (
            f"alpha={alpha} is not configured for this CQR calibrator; "
            f"configured alphas are {self.alphas}."
        )
        raise ValueError(msg)

    def _has_alpha_dimension(self, predictions: Tensor) -> bool:
        return len(self.alphas) > 1 and predictions.shape[-1] == len(self.alphas)

    def _select_alpha(self, predictions: Tensor, alpha_idx: int) -> Tensor:
        if len(self.alphas) == 1:
            return predictions
        return predictions[..., alpha_idx]
