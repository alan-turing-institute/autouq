"""Axis-grouping, standardisation, and validation helpers shared by calibrators.

These operate on the working layout ``(B, T, *S, C)`` -- the forecast with its
ensemble-member axis already reduced. Batch (axis 0) is always pooled; each
calibrator reduces the member axis itself before calling these. Keeping the
axis-grouping, standardisation and validation logic here lets the marginal and
conformal families share one implementation rather than re-deriving it.
"""

import math
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


def normalize_roles(per: Sequence[AxisRole | str]) -> tuple[AxisRole, ...]:
    """Convert axis-role names to :class:`AxisRole` members.

    Bare strings are accepted as a ``StrEnum`` convenience, but everything
    downstream works with the enum members, so normalise at this single boundary
    and validate eagerly.

    Args:
        per: Axis roles as :class:`AxisRole` members or their string values.

    Returns:
        The roles as :class:`AxisRole` members, in the given order.
    """
    roles: list[AxisRole] = []
    for name in per:
        try:
            roles.append(AxisRole(name))
        except ValueError:
            valid = ", ".join(repr(r.value) for r in AxisRole)
            msg = f"unknown axis role {name!r}; expected one of {valid}."
            raise ValueError(msg) from None
    return tuple(roles)


def resolve_group_dims(ndim: int, per: Sequence[AxisRole | str]) -> tuple[int, ...]:
    """Map axis roles to dimension indices of a ``(B, T, *S, C)`` tensor.

    Args:
        ndim: Rank of the working tensor (ensemble members already reduced).
        per: Axis roles that get their own coefficients, as :class:`AxisRole`
            members or their string values. ``SPACE`` expands to every spatial
            dimension.

    Returns:
        Independent dimension indices, sorted ascending. Batch (axis 0) is never
        included -- it is always pooled.
    """
    # channels-last (B, T, *S, C) convention -- matches the ``TensorBTSC*``
    # aliases in ``autouq.types``.
    role_to_dims = {
        AxisRole.TIME: (1,),
        AxisRole.SPACE: tuple(range(2, ndim - 1)),
        AxisRole.CHANNEL: (ndim - 1,),
    }
    dims: list[int] = []
    for role in normalize_roles(per):
        dims.extend(role_to_dims[role])
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


def group_layout(
    shape: Sequence[int], per: Sequence[AxisRole | str]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Group dims, coefficient shape and broadcast view for ``per``.

    Bundles :func:`resolve_group_dims`, :func:`group_param_shape` and
    :func:`broadcast_view` -- the three are always used together to lay
    per-group coefficients over a working ``(B, T, *S, C)`` tensor.

    Args:
        shape: Shape of the working tensor (ensemble members already reduced).
        per: Axis roles that get their own coefficients, as :class:`AxisRole`
            members or their string values.

    Returns:
        ``(group_dims, param_shape, broadcast_view)``.
    """
    ndim = len(shape)
    group_dims = resolve_group_dims(ndim, per)
    param_shape = group_param_shape(shape, group_dims)
    view = broadcast_view(ndim, group_dims, param_shape)
    return group_dims, param_shape, view


def group_location_scale(
    y: TensorBTSC, group_dims: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """Per-group location (mean) and scale (floored std) of ``y``.

    Reduces over every axis not in ``group_dims`` (batch and the pooled
    structural axes), leaving one statistic per independent-axis combination.
    """
    red = pooled_dims(y.ndim, group_dims)
    n_pooled = math.prod(y.shape[d] for d in red)
    if n_pooled < 2:
        msg = (
            "could not estimate a per-group scale from `true`; each group needs "
            f"at least 2 pooled samples but got {n_pooled} "
            "(check the batch size and `per`)."
        )
        raise ValueError(msg)
    loc = y.mean(dim=red)
    scale = y.std(dim=red).clamp_min(_SCALE_FLOOR)
    if not torch.isfinite(scale).all():
        msg = "per-group scale is non-finite; check `true` for NaN/inf values."
        raise ValueError(msg)
    return loc, scale


def validate_alpha(alpha: float) -> None:
    """Require a single miscoverage level in the open interval (0, 1)."""
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie in the open interval (0, 1); got {alpha}."
        raise ValueError(msg)


def validate_alphas(alphas: float | Sequence[float]) -> list[float]:
    """Normalise ``alphas`` to a non-empty list, each in the open interval (0, 1).

    A bare ``int`` is rejected with :class:`TypeError`: a miscoverage level is a
    float in ``(0, 1)``, so an integer alpha is always a mistake -- ``0`` and
    ``1`` are excluded and nothing lies strictly between them.
    """
    if isinstance(alphas, float):
        values = [float(alphas)]
    elif isinstance(alphas, int):
        msg = "alpha must be a float or a sequence of floats."
        raise TypeError(msg)
    else:
        values = [float(a) for a in alphas]
    if not values:
        msg = "at least one alpha is required."
        raise ValueError(msg)
    for alpha in values:
        validate_alpha(alpha)
    return values
