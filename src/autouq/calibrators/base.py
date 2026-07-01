import abc
from collections.abc import Sequence

from autouq.types import Tensor


class Calibrator(abc.ABC):
    """Calibrator base class."""

    def __init__(self, spatial_dims: Sequence[int]):
        # TODO: consider spatial handling
        self.spatial_dims = spatial_dims

    @abc.abstractmethod
    def calibrate(self, y_true: Tensor, y_pred: Tensor) -> None: ...
