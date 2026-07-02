import math
from collections.abc import Sequence

import torch

from autouq.calibrators.conformal.conformal import ConformalCalibrator
from autouq.types import (
    TensorBNC,
    TensorBNCK,
    TensorBNCQ,
    TensorBNCQK,
    TensorBNIA,
    TensorNC,
    TensorNCK,
)


class ConformalizedQuantileRegression(
    ConformalCalibrator[
        TensorBNCQ | TensorBNCQK,
        TensorBNC | TensorBNCK,
        TensorNC | TensorNCK,
    ]
):
    r"""Conformalized quantile regression calibrator.

    Calibration scores are computed cell-wise from lower and upper quantile
    pred tensors:

    .. math::

        s_i = \max(\hat{q}_{\mathrm{lo}}(x_i) - y_i,
        y_i - \hat{q}_{\mathrm{hi}}(x_i)).

    Prediction intervals expand the quantile pair by the conformal score
    threshold:

    .. math::

        [\hat{q}_{\mathrm{lo}}(x) - \hat{s}_{1-\alpha},
         \hat{q}_{\mathrm{hi}}(x) + \hat{s}_{1-\alpha}].

    Args:
        quantile_level_pairs: Lower/upper quantile-level pairs. For example,
            ``[(0.05, 0.95), (0.1, 0.9)]`` defines pairs for prediction
            miscoverage levels ``[0.1, 0.2]``.
        temporal_dim: Optional index of the temporal dimension in tensors passed
            to the calibrator.
        spatial_dims: Optional indices of spatial dimensions in tensors passed
            to the calibrator.
    """

    def __init__(
        self,
        quantile_level_pairs: Sequence[tuple[float, float]],
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        super().__init__(temporal_dim=temporal_dim, spatial_dims=spatial_dims)
        self.quantile_level_pairs, self.quantile_pair_alphas = (
            self._normalize_quantile_level_pairs(quantile_level_pairs)
        )

    def _score(
        self,
        true: TensorBNC,
        pred: TensorBNCQ | TensorBNCQK,
    ) -> TensorBNC | TensorBNCK:
        lower_quantile, upper_quantile = self._split_quantile_preds(pred)
        target_shape = (
            lower_quantile.shape[:-1]
            if self._has_quantile_pair_dimension(lower_quantile)
            else lower_quantile.shape
        )
        if true.shape != target_shape:
            msg = (
                "true must match the quantile pred shape; "
                f"got {tuple(true.shape)} and expected {tuple(target_shape)}."
            )
            raise ValueError(msg)
        self._validate_quantile_order(lower_quantile, upper_quantile)
        if self._has_quantile_pair_dimension(lower_quantile):
            true = true.unsqueeze(-1)
        return torch.maximum(lower_quantile - true, true - upper_quantile)

    def score_quantile(self, alpha: float) -> TensorNC:
        quantile_pair_idx = self._quantile_pair_index(alpha)
        score_quantile = super().score_quantile(alpha)
        if len(self.quantile_pair_alphas) == 1:
            return score_quantile
        return score_quantile[..., quantile_pair_idx]

    def _predict(
        self,
        pred: TensorBNCQ | TensorBNCQK,
        alphas: Sequence[float],
    ) -> TensorBNIA:
        scores = self._calibration_scores()
        lower_quantiles, upper_quantiles = self._split_quantile_preds(pred)
        if lower_quantiles.shape[1:] != scores.shape[1:]:
            msg = (
                "pred must match the calibrated trailing shape; "
                f"got {tuple(lower_quantiles.shape[1:])} and expected "
                f"{tuple(scores.shape[1:])}."
            )
            raise ValueError(msg)
        self._validate_quantile_order(lower_quantiles, upper_quantiles)

        intervals = []
        for alpha in alphas:
            quantile_pair_idx = self._quantile_pair_index(alpha)
            lower_quantile = self._select_quantile_pair(
                lower_quantiles,
                quantile_pair_idx,
            )
            upper_quantile = self._select_quantile_pair(
                upper_quantiles,
                quantile_pair_idx,
            )
            score_quantile = self.score_quantile(alpha).to(device=pred.device)
            intervals.append(
                torch.stack(
                    (
                        lower_quantile - score_quantile,
                        upper_quantile + score_quantile,
                    ),
                    dim=-1,
                )
            )
        return torch.stack(intervals, dim=-1)

    def _split_quantile_preds(
        self,
        pred: TensorBNCQ | TensorBNCQK,
    ) -> tuple[TensorBNC | TensorBNCK, TensorBNC | TensorBNCK]:
        if len(self.quantile_pair_alphas) == 1:
            if pred.ndim < 2 or pred.shape[-1] != 2:
                msg = (
                    "CQR pred tensors for one quantile pair must include a "
                    f"final lower/upper quantile dimension of size 2; got shape "
                    f"{tuple(pred.shape)}."
                )
                raise ValueError(msg)
            return pred[..., 0], pred[..., 1]

        if pred.ndim < 3 or pred.shape[-2] != 2:
            msg = (
                "CQR pred tensors for multiple quantile pairs must include a "
                f"lower/upper quantile dimension of size 2 before the "
                f"quantile_pairs dimension; "
                f"got shape {tuple(pred.shape)}."
            )
            raise ValueError(msg)
        if pred.shape[-1] != len(self.quantile_pair_alphas):
            msg = (
                "CQR pred tensors must include one lower/upper quantile pair per "
                f"configured quantile_level_pair; got {pred.shape[-1]} "
                f"quantile_pair entries and expected "
                f"{len(self.quantile_pair_alphas)} from "
                f"{len(self.quantile_level_pairs)} quantile_level_pairs."
            )
            raise ValueError(msg)
        return pred[..., 0, :], pred[..., 1, :]

    def _normalize_quantile_level_pairs(
        self,
        quantile_level_pairs: Sequence[tuple[float, float]],
    ) -> tuple[list[tuple[float, float]], list[float]]:
        if not quantile_level_pairs:
            msg = "At least one quantile_level_pair is required."
            raise ValueError(msg)

        normalized_pairs = [
            (float(lower_level), float(upper_level))
            for lower_level, upper_level in quantile_level_pairs
        ]
        for lower_level, upper_level in normalized_pairs:
            self._validate_quantile_level(lower_level)
            self._validate_quantile_level(upper_level)
            if lower_level >= upper_level:
                msg = (
                    "quantile_level_pairs must be ordered as "
                    f"(lower, upper); got {(lower_level, upper_level)}."
                )
                raise ValueError(msg)

        quantile_pair_alphas = [
            lower_level + (1 - upper_level)
            for lower_level, upper_level in normalized_pairs
        ]
        for idx, quantile_pair_alpha in enumerate(quantile_pair_alphas):
            if any(
                math.isclose(
                    quantile_pair_alpha,
                    other_quantile_pair_alpha,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for other_quantile_pair_alpha in quantile_pair_alphas[idx + 1 :]
            ):
                msg = "quantile_level_pairs must define distinct alphas."
                raise ValueError(msg)
        return normalized_pairs, quantile_pair_alphas

    def _validate_quantile_level(self, level: float) -> None:
        if not 0 < level < 1:
            msg = f"quantile level must be between 0 and 1, got {level}."
            raise ValueError(msg)

    def _validate_quantile_order(
        self,
        lower_quantile: TensorBNC | TensorBNCK,
        upper_quantile: TensorBNC | TensorBNCK,
    ) -> None:
        if torch.any(lower_quantile > upper_quantile):
            msg = "Lower quantile pred tensors must not exceed upper pred tensors."
            raise ValueError(msg)

    def _quantile_pair_index(self, alpha: float) -> int:
        self._validate_alpha(alpha)
        for idx, configured_alpha in enumerate(self.quantile_pair_alphas):
            if math.isclose(alpha, configured_alpha, rel_tol=1e-12, abs_tol=1e-12):
                return idx
        msg = (
            f"alpha={alpha} does not match any CQR quantile pair; "
            f"quantile_level_pairs={self.quantile_level_pairs} define valid "
            f"alphas {self.quantile_pair_alphas}."
        )
        raise ValueError(msg)

    def _has_quantile_pair_dimension(
        self,
        quantile_pred_values: TensorBNC | TensorBNCK,
    ) -> bool:
        return len(self.quantile_pair_alphas) > 1 and quantile_pred_values.shape[
            -1
        ] == len(self.quantile_pair_alphas)

    def _select_quantile_pair(
        self,
        quantile_pred_values: TensorBNC | TensorBNCK,
        quantile_pair_idx: int,
    ) -> TensorBNC:
        if len(self.quantile_pair_alphas) == 1:
            return quantile_pred_values
        return quantile_pred_values[..., quantile_pair_idx]
