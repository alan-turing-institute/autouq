from typing import Protocol, runtime_checkable

import torch

from autouq.types import Tensor, TensorB, TensorBNC, TensorBNI


@runtime_checkable
class RiskLoss(Protocol):
    """Loss contract used by risk-controlling prediction sets.

    Implementations return one finite, non-negative loss per independent
    calibration example. Loss must be non-increasing as prediction sets grow.
    """

    def __call__(self, prediction_set: Tensor, true: Tensor) -> TensorB:
        """Return one loss for each item on the batch dimension."""


class CoverageLoss:
    r"""Per-example miscoverage fraction for interval-valued predictions.

    Despite the short public name, this is the loss associated with coverage:
    one minus the fraction of cells whose interval contains the true value.
    For a prediction set :math:`T(X)` with cells indexed by :math:`j`, it is

    .. math::

        L(T(X), Y) = \frac{1}{J}\sum_j \mathbb{1}\{Y_j \notin T(X)_j\}.

    The output has shape ``(batch,)`` and therefore treats calibration examples,
    rather than individual temporal or spatial cells, as the independent draws.
    """

    def __call__(
        self,
        prediction_set: TensorBNI,
        true: TensorBNC,
    ) -> TensorB:
        if prediction_set.shape != (*true.shape, 2):
            msg = (
                "prediction_set must have true.shape followed by a lower/upper "
                f"axis of size 2; got {tuple(prediction_set.shape)} and "
                f"{tuple(true.shape)}."
            )
            raise ValueError(msg)

        lower, upper = prediction_set.unbind(dim=-1)
        covered = (lower <= true) & (true <= upper)
        uncovered = torch.logical_not(covered).reshape(true.shape[0], -1)
        return uncovered.to(dtype=prediction_set.dtype).mean(dim=1)
