import abc
from collections.abc import Sequence
from typing import Generic, TypeVar

from autouq.types import Tensor, TensorBNC

PredT = TypeVar("PredT", bound=Tensor)


class Calibrator(Generic[PredT], abc.ABC):
    """Base class for prediction calibrators.

    Args:
        temporal_dim: Optional index of the temporal dimension in tensors passed
            to the calibrator.
        spatial_dims: Optional indices of spatial dimensions in tensors passed
            to the calibrator.
    """

    def __init__(
        self,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        self.temporal_dim = temporal_dim
        self.spatial_dims = spatial_dims

    @abc.abstractmethod
    def calibrate(self, true: TensorBNC, pred: PredT) -> None: ...
