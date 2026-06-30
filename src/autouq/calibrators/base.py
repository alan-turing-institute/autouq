import abc
from collections.abc import Sequence

import torch

from autouq.types import Tensor


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
