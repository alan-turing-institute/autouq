import abc
import math
from collections.abc import Sequence
from typing import Generic, TypeVar, cast

from autouq.calibrators.base import Calibrator, PredT
from autouq.calibrators.grouping import validate_alpha, validate_alphas
from autouq.types import Tensor, TensorBN, TensorBNC, TensorBNIA, TensorN

ScoreT = TypeVar("ScoreT", bound=TensorBN)
ScoreQuantileT = TypeVar("ScoreQuantileT", bound=TensorN)


class ConformalCalibrator(
    Calibrator[PredT],
    Generic[PredT, ScoreT, ScoreQuantileT],
    abc.ABC,
):
    """Base class for split-conformal calibrators.

    Args:
        temporal_dim: Optional index of the temporal dimension in tensors passed
            to the calibrator.
        spatial_dims: Optional indices of spatial dimensions in tensors passed
            to the calibrator.

    Attributes:
        scores: Cached calibration scores with the calibration examples on
            dimension 0.
    """

    def __init__(
        self,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        super().__init__(temporal_dim=temporal_dim, spatial_dims=spatial_dims)
        self.scores: ScoreT | None = None

    @abc.abstractmethod
    def _score(self, true: TensorBNC, pred: PredT) -> ScoreT: ...

    def cache_scores(self, scores: ScoreT) -> None:
        if scores.ndim == 0:
            msg = "Calibration scores must include a calibration dimension."
            raise ValueError(msg)
        if scores.shape[0] == 0:
            msg = "Calibration scores must contain at least one sample."
            raise ValueError(msg)
        self.scores = scores

    def calibrate(self, true: TensorBNC, pred: PredT) -> None:
        self.cache_scores(self._score(true, pred))

    def _calibration_scores(self) -> ScoreT:
        if self.scores is None:
            msg = "Calibrator must be calibrated before computing the score quantile."
            raise RuntimeError(msg)
        return self.scores

    def score_quantile(self, alpha: float) -> ScoreQuantileT:
        r"""Return the conformal score threshold.

        This is the finite-sample empirical quantile of calibration scores,
        often denoted :math:`\hat{q}_{1-\alpha}` in split conformal prediction.
        For :math:`n` calibration examples, this uses the one-indexed order
        statistic

        .. math::

            k = \lceil (n + 1)(1 - \alpha) \rceil.

        Args:
            alpha: Miscoverage level in ``(0, 1)``.

        Returns:
            Score threshold with the calibration dimension removed.
        """
        validate_alpha(alpha)

        scores = self._calibration_scores()
        n_calibration = scores.shape[0]
        order = math.ceil((n_calibration + 1) * (1 - alpha))
        if order > n_calibration:
            min_alpha = 1 / (n_calibration + 1)
            msg = (
                f"alpha={alpha} is too small for {n_calibration} calibration "
                f"samples; use alpha >= {min_alpha:g}."
            )
            raise ValueError(msg)
        return cast("ScoreQuantileT", scores.kthvalue(order, dim=0).values)

    @abc.abstractmethod
    def _predict(self, pred: PredT, alphas: Sequence[float]) -> TensorBNIA: ...

    def predict(self, pred: PredT, alphas: float | Sequence[float]) -> TensorBNIA:
        alpha_values = validate_alphas(alphas)
        # TODO: consider time dimension regarding the exchangeability assumption
        # should time be in batch dim ?
        # (see 2.4: https://arxiv.org/abs/2408.09881)
        # - e.g. could have helper functions for splitting into chunks
        # - also could have validation methods for checking the assumption
        return self._predict(pred, alpha_values)


class StandardDeviation(ConformalCalibrator[Tensor, TensorBN, TensorN]):
    """Standard Deviation base class."""
