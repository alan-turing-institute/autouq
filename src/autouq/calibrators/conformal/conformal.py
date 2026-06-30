import abc
import math
from collections.abc import Callable, Sequence

from autouq.calibrators.base import Calibrator
from autouq.types import Tensor, TensorBTSIA


class ConformalCalibrator(Calibrator, abc.ABC):
    """Conformal calibrator base class."""

    def __init__(self, spatial_dims: Sequence[int]):
        super().__init__(spatial_dims)
        self.scores: Tensor | None = None

    @abc.abstractmethod
    def _score_fn(self) -> Callable: ...

    def cache_scores(self, scores: Tensor):
        if scores.ndim == 0:
            msg = "Calibration scores must include a calibration dimension."
            raise ValueError(msg)
        if scores.shape[0] == 0:
            msg = "Calibration scores must contain at least one sample."
            raise ValueError(msg)
        self.scores = scores

    def calibrate(self, y_true: Tensor, y_pred: Tensor):
        scores = self._score_fn()(y_true, y_pred)
        self.cache_scores(scores)

    def _calibration_scores(self) -> Tensor:
        if self.scores is None:
            msg = "Calibrator must be calibrated before computing q_hat."
            raise RuntimeError(msg)
        return self.scores

    def q_hat(self, alpha: float) -> Tensor:
        if not 0 < alpha < 1:
            msg = f"alpha must be between 0 and 1, got {alpha}."
            raise ValueError(msg)

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
    def _predict(self, y_pred: Tensor, alphas: Sequence[float]) -> TensorBTSIA: ...

    def predict(self, y_pred: Tensor, alphas: float | Sequence[float]) -> TensorBTSIA:
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
        # TODO: consider time dimension regarding the exchangeability assumption
        # should time be in batch dim ?
        # (see 2.4: https://arxiv.org/abs/2408.09881)
        # - e.g. could have helper functions for splitting into chunks
        # - also could have validation methods for checking the assumption
        return self._predict(y_pred, alpha_values)


class ConformalizedQuantileRegression(ConformalCalibrator):
    """Conformalized Quantile Regression base class."""


class StandardDeviation(ConformalCalibrator):
    """Standard Deviation base class."""


class Ensemble(ConformalCalibrator):
    """Ensemble base class."""
