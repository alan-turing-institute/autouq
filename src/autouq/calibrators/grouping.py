"""Axis-grouping, standardisation, and validation helpers shared by calibrators.

These operate on the working layout ``(B, T, *S, C)`` -- the forecast with its
ensemble-member axis already reduced. Batch (axis 0) is always pooled; each
calibrator reduces the member axis itself before calling these. Keeping the
axis-grouping, standardisation and validation logic here lets the marginal and
conformal families share one implementation rather than re-deriving it.
"""

from collections.abc import Sequence
from enum import StrEnum

import torch

from autouq.types import Tensor, TensorBTSC

_SCALE_FLOOR = 1e-8


class AxisRole(StrEnum):
    """Forecast axis roles that can each receive their own coefficients.

    Members are plain strings (``StrEnum``), so ``AxisRole.TIME`` and ``"time"``
    are interchangeable wherever an axis role is accepted.
    """

    TIME = "time"
    SPACE = "space"
    CHANNEL = "channel"


def resolve_group_dims(ndim: int, per: Sequence[AxisRole | str]) -> tuple[int, ...]:
    """Map axis-role names to dimension indices of a ``(B, T, *S, C)`` tensor.

    Parameters
    ----------
    ndim
        Rank of the working tensor (ensemble members already reduced).
    per
        Axis roles that get their own coefficients, as :class:`AxisRole` members
        or their string values. ``SPACE`` expands to every spatial dimension.

    Returns
    -------
    tuple of int
        Independent dimension indices, sorted ascending. Batch (axis 0) is never
        included -- it is always pooled.
    """
    # channels-last (B, T, *S, C) convention -- matches the ``TensorBTSC*``
    # aliases in ``autouq.types``.
    roles = {
        AxisRole.TIME: (1,),
        AxisRole.SPACE: tuple(range(2, ndim - 1)),
        AxisRole.CHANNEL: (ndim - 1,),
    }
    dims: list[int] = []
    for name in per:
        try:
            role = AxisRole(name)
        except ValueError:
            valid = ", ".join(repr(r.value) for r in AxisRole)
            msg = f"unknown axis role {name!r}; expected one of {valid}."
            raise ValueError(msg) from None
        dims.extend(roles[role])
    return tuple(sorted(set(dims)))


def pooled_dims(ndim: int, group_dims: Sequence[int]) -> tuple[int, ...]:
    """Axes to reduce over: every dimension not in ``group_dims``."""
    grp = set(group_dims)
    return tuple(d for d in range(ndim) if d not in grp)


def group_param_shape(
    working_shape: Sequence[int], group_dims: Sequence[int]
) -> tuple[int, ...]:
    """Coefficient shape for ``group_dims``: the working sizes on those axes."""
    return tuple(working_shape[d] for d in group_dims)


def broadcast_view(
    ndim: int, group_dims: Sequence[int], param_shape: Sequence[int]
) -> tuple[int, ...]:
    """View placing per-group coefficients on their axes, ``1`` on pooled axes."""
    view = [1] * ndim
    for d, size in zip(group_dims, param_shape, strict=True):
        view[d] = size
    return tuple(view)


def group_location_scale(
    y: TensorBTSC, group_dims: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """Per-group location (mean) and scale (floored std) of ``y``.

    Reduces over every axis not in ``group_dims`` (batch and the pooled
    structural axes), leaving one statistic per independent-axis combination.
    """
    red = pooled_dims(y.ndim, group_dims)
    loc = y.mean(dim=red)
    scale = y.std(dim=red).clamp_min(_SCALE_FLOOR)
    if not torch.isfinite(scale).all():
        msg = (
            "could not estimate a finite per-group scale from y_true; each group "
            "needs at least 2 pooled samples (check the batch size and `per`)."
        )
        raise ValueError(msg)
    return loc, scale


def validate_alphas(alphas: float | Sequence[float]) -> list[float]:
    """Normalise ``alphas`` to a list, requiring each in the open interval (0, 1)."""
    values = (
        [float(alphas)]
        if isinstance(alphas, int | float)
        else [float(a) for a in alphas]
    )
    if any(not 0.0 < a < 1.0 for a in values):
        msg = f"every alpha must lie in the open interval (0, 1); got {values}."
        raise ValueError(msg)
    return values
