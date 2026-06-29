import abc
from collections.abc import Callable, Sequence

import torch

from autouq.types import Tensor, TensorBTSIA


# Example Algorithm
# 1. Fit Model to training data
# .   - No API needed
# 2. Fit calibration data x_c
#    - Get predictions (e.g. means): f(x_c). No API needed, entry point
#    - calculate absolute residuals: r_c = |y_c - f(x_c)|
#    - For a given alpha (target coverage) calculate
#           q = np.ceil((n+1)*(1-alpha))/n quantile of {r_c}'s
#      where n = len(r_c).
# 3. Get conformalized predictions
#     - calculate f(x_t), y_pred_t
#     - conformalize(y_pred_t) = y_pred_t ± q
class Calibrator(abc.ABC):
    """Calibrator base class."""

    def __init__(self, spatial_dims: Sequence[int]):
        # TODO: consider spatial handling
        self.spatial_dims = spatial_dims

    @abc.abstractmethod
    def calibrate(self, y_true: Tensor, y_pred: Tensor): ...

    # TODO: what should we have for predict here?
    def predict(self, y_pred: Tensor, alphas: float | Sequence[float]) -> Tensor:  # noqa: ARG002
        # TODO: replace with impl
        return torch.tensor(0)


class ConformalCalibrator(Calibrator, abc.ABC):
    """Conformal calibrator base class."""

    @abc.abstractmethod
    def _score_fn(self) -> Callable: ...

    def cache_scores(self, scores: Tensor):
        self.scores = scores

    def calibrate(self, y_true: Tensor, y_pred: Tensor):
        scores = self._score_fn()(y_true, y_pred)
        self.cache_scores(scores)

    def q_hat(self, alpha: float) -> Tensor:  # noqa: ARG002
        # TODO: add impl, placeholder
        return torch.tensor(0)

    @abc.abstractmethod
    def _predict(
        self, y_pred: Tensor, alpha: float | Sequence[float]
    ) -> TensorBTSIA:  # (B, T, *S, C, 2, n_alphas)?
        pass

    def predict(
        self, y_pred: Tensor, alphas: float | Sequence[float]
    ) -> TensorBTSIA:  # (B, T, *S, C, 2, n_alphas)?
        alphas = [alphas] if isinstance(alphas, float) else alphas
        # TODO: consider time dimension regarding the exchangeability assumption
        # should time be in batch dim ?
        # (see 2.4: https://arxiv.org/abs/2408.09881)
        # - e.g. could have helper functions for splitting into chunks
        # - also could have validation methods for checking the assumption
        return self._predict(y_pred, alphas)


# Target
class ConformalizedQuantileRegression(ConformalCalibrator):
    """Conformalized Quantile Regression base class."""


class AbsoluteErrorResidual(ConformalCalibrator):
    """Absolute Error Residual base class."""

    def _score_fn(self) -> Callable:
        # TODO: Calculate absolute residuals: r_c = |y_c - f(x_c)|
        return lambda _: _


class StandardDeviation(ConformalCalibrator):
    """Standard Deviation base class."""


class Ensemble(ConformalCalibrator):
    """Ensemble base class."""
