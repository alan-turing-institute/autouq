import abc
import math
from collections.abc import Sequence
from typing import Generic

from autouq.calibrators.base import Calibrator, PredT
from autouq.types import Tensor, TensorBNC, TensorBNIA


class ConformalCalibrator(Calibrator[PredT], Generic[PredT], abc.ABC):
    """Conformal calibrator base class."""

    def __init__(
        self,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        super().__init__(temporal_dim=temporal_dim, spatial_dims=spatial_dims)
        self.scores: Tensor | None = None

    @abc.abstractmethod
    def _score(self, true: TensorBNC, pred: PredT) -> Tensor: ...

    def cache_scores(self, scores: Tensor) -> None:
        if scores.ndim == 0:
            msg = "Calibration scores must include a calibration dimension."
            raise ValueError(msg)
        if scores.shape[0] == 0:
            msg = "Calibration scores must contain at least one sample."
            raise ValueError(msg)
        self.scores = scores

    def _normalize_alphas(self, alphas: float | Sequence[float]) -> list[float]:
        if isinstance(alphas, float):
            alpha_values = [float(alphas)]
        elif isinstance(alphas, int):
            msg = "alpha must be a float or a sequence of floats."
            raise TypeError(msg)
        else:
            alpha_values = [float(alpha) for alpha in alphas]

        if not alpha_values:
            msg = "At least one alpha is required."
            raise ValueError(msg)
        for alpha in alpha_values:
            self._validate_alpha(alpha)
        return alpha_values

    def calibrate(self, true: TensorBNC, pred: PredT) -> None:
        self.cache_scores(self._score(true, pred))

    def _calibration_scores(self) -> Tensor:
        if self.scores is None:
            msg = "Calibrator must be calibrated before computing the score quantile."
            raise RuntimeError(msg)
        return self.scores

    def _validate_alpha(self, alpha: float) -> None:
        if not 0 < alpha < 1:
            msg = f"alpha must be between 0 and 1, got {alpha}."
            raise ValueError(msg)

    def score_quantile(self, alpha: float) -> Tensor:
        """Return the conformal score threshold.

        This is the finite-sample empirical quantile of calibration scores,
        often denoted Q_{1-alpha} or q_hat in split conformal prediction.
        """
        self._validate_alpha(alpha)

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
        return scores.kthvalue(order, dim=0).values

    @abc.abstractmethod
    def _predict(self, pred: PredT, alphas: Sequence[float]) -> TensorBNIA: ...

    def predict(self, pred: PredT, alphas: float | Sequence[float]) -> TensorBNIA:
        alpha_values = self._normalize_alphas(alphas)
        # TODO: consider time dimension regarding the exchangeability assumption
        # should time be in batch dim ?
        # (see 2.4: https://arxiv.org/abs/2408.09881)
        # - e.g. could have helper functions for splitting into chunks
        # - also could have validation methods for checking the assumption
        return self._predict(pred, alpha_values)


class ConformalizedQuantileRegression(ConformalCalibrator[Tensor]):
    """Conformalized Quantile Regression base class."""


class StandardDeviation(ConformalCalibrator[Tensor]):
    """Standard Deviation base class."""


class Ensemble(ConformalCalibrator[Tensor]):
    """Ensemble base class."""
