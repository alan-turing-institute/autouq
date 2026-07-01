import abc
from collections.abc import Sequence
from typing import Generic, TypeVar

from autouq.types import Tensor, TensorBNC

PredT = TypeVar("PredT", bound=Tensor)


class Calibrator(Generic[PredT], abc.ABC):
    """Calibrator base class."""

    def __init__(
        self,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        self.temporal_dim = temporal_dim
        self.spatial_dims = spatial_dims

    @abc.abstractmethod
    def calibrate(self, true: TensorBNC, pred: PredT) -> None: ...
