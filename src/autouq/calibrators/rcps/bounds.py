import math
from typing import Protocol, runtime_checkable

from autouq.calibrators.grouping import validate_open_unit_interval
from autouq.types import TensorB


@runtime_checkable
class RiskBound(Protocol):
    """Upper-confidence-bound contract used by RCPS calibration.

    Bounds used by the binary search must preserve the ordering of
    non-increasing per-example losses.
    """

    def __call__(self, losses: TensorB, delta: float) -> float:
        """Bound population risk from per-example calibration losses."""


class HoeffdingBound:
    """Hoeffding upper confidence bound for losses in ``[0, 1]``."""

    def __call__(self, losses: TensorB, delta: float) -> float:
        validate_open_unit_interval(delta, name="delta")
        if ((losses < 0) | (losses > 1)).any():
            msg = "HoeffdingBound requires loss values in the interval [0, 1]."
            raise ValueError(msg)
        margin = math.sqrt(math.log(1 / delta) / (2 * losses.shape[0]))
        return float(losses.mean().item() + margin)
