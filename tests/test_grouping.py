import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from autouq.calibrators.grouping import (
    AxisRole,
    group_layout,
    group_location_scale,
    normalize_roles,
    validate_alpha,
    validate_alphas,
)


def test_validate_alpha_accepts_interior_point():
    assert validate_alpha(0.5) is None


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_validate_alpha_rejects_endpoints_and_outside(bad):
    with pytest.raises(ValueError, match="open interval"):
        validate_alpha(bad)


def test_validate_alphas_normalises_scalar_and_sequence():
    assert validate_alphas(0.1) == [0.1]
    assert validate_alphas([0.1, 0.2]) == [0.1, 0.2]
    assert validate_alphas((0.05,)) == [0.05]


def test_validate_alphas_rejects_bare_int():
    # A miscoverage level is a float in (0, 1); an int alpha is always a mistake.
    # With runtime typechecking on (tests/CI) beartype rejects it at the
    # signature; with it off, the body's own guard raises TypeError. Both hold.
    with pytest.raises((TypeError, BeartypeCallHintParamViolation)):
        validate_alphas(1)


def test_validate_alphas_rejects_empty():
    with pytest.raises(ValueError, match="at least one alpha"):
        validate_alphas([])


def test_validate_alphas_rejects_out_of_range_member():
    with pytest.raises(ValueError, match="open interval"):
        validate_alphas([0.1, 0.0])


def test_normalize_roles_maps_strings_and_members_to_enum():
    # both the string convenience form and the enum itself normalise to members,
    # preserving order; downstream logic then only ever sees AxisRole members.
    out = normalize_roles(("time", AxisRole.SPACE, "channel"))
    assert out == (AxisRole.TIME, AxisRole.SPACE, AxisRole.CHANNEL)
    assert all(isinstance(role, AxisRole) for role in out)


def test_normalize_roles_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown axis role"):
        normalize_roles(("altitude",))


@pytest.mark.parametrize(
    ("per", "group_dims", "param_shape", "view"),
    [
        ((AxisRole.TIME,), (1,), (3,), (1, 3, 1, 1)),
        ((AxisRole.SPACE,), (2,), (4,), (1, 1, 4, 1)),
        ((AxisRole.CHANNEL,), (3,), (1,), (1, 1, 1, 1)),
        ((AxisRole.TIME, AxisRole.SPACE), (1, 2), (3, 4), (1, 3, 4, 1)),
    ],
)
def test_group_layout_bundles_dims_shape_and_view(per, group_dims, param_shape, view):
    # working (B, T, S, C) = (50, 3, 4, 1)
    assert group_layout((50, 3, 4, 1), per) == (group_dims, param_shape, view)


def test_group_location_scale_reduces_pooled_axes():
    # per=("time",) on (B, T, S, C): pool over batch/space/channel, keep time
    y = torch.randn(50, 3, 4, 1)
    loc, scale = group_location_scale(y, group_dims=(1,))
    assert tuple(loc.shape) == (3,)
    assert tuple(scale.shape) == (3,)
    assert torch.all(scale > 0)


def test_group_location_scale_rejects_degenerate_group():
    # only batch is pooled and batch size is 1: a single sample per group has no
    # estimable spread, so the scale is non-finite and must be rejected clearly.
    y = torch.randn(1, 2, 1, 1)
    with pytest.raises(ValueError, match="per-group scale"):
        group_location_scale(y, group_dims=(1, 2, 3))
