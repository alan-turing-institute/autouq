import pytest
import torch

from autouq.calibrators.ecc import ECC


def test_apply_preserves_each_marginal_exactly():
    torch.manual_seed(0)
    samples = torch.randn(5, 4)  # (members, sites), member_dim=0
    template = torch.randn(5, 4)
    out = ECC(member_dim=0).apply(samples, template)
    # each site's set of values is unchanged; only the order across members
    assert torch.allclose(out.sort(dim=0).values, samples.sort(dim=0).values)


def test_apply_imposes_template_rank_order():
    samples = torch.tensor([[10.0], [20.0], [30.0]])  # 3 members, 1 site
    template = torch.tensor([[5.0], [1.0], [9.0]])  # ranks: 1, 0, 2
    out = ECC(member_dim=0).apply(samples, template)
    # member with the largest template value gets the largest marginal value
    assert torch.equal(out[:, 0], torch.tensor([20.0, 10.0, 30.0]))


def test_apply_with_self_template_is_identity():
    torch.manual_seed(0)
    samples = torch.randn(6, 3)
    out = ECC(member_dim=0).apply(samples, samples)
    assert torch.allclose(out, samples)


def test_apply_last_axis_members_btscm_shape():
    torch.manual_seed(0)
    samples = torch.randn(2, 1, 3, 1, 5)  # (B, T, S, C, M)
    template = torch.randn(2, 1, 3, 1, 5)
    out = ECC().apply(samples, template)  # member_dim=-1 default
    assert out.shape == samples.shape
    assert torch.allclose(out.sort(dim=-1).values, samples.sort(dim=-1).values)


def test_apply_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        ECC(member_dim=0).apply(torch.randn(5, 4), torch.randn(5, 3))


def test_apply_rejects_too_few_members():
    with pytest.raises(ValueError, match="at least 2 members"):
        ECC(member_dim=0).apply(torch.randn(1, 4), torch.randn(1, 4))
