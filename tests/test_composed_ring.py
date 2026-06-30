"""Acceptance test: EMOS + ECC on a correlated-Gaussian ring.

Mirrors the multistep-rollout report's ensemble-copula-coupling demonstration. A
spatially-white "diagonal" model has the correct per-site marginal but no
cross-site dependence, so the regional total is badly under-covered. EMOS leaves
the marginals (already correct) alone; ECC restores the joint coverage -- but
only when reordered onto a template that carries the true dependence.
"""

import torch

from autouq.calibrators import EMOSECC


def _ring_cov(n_sites, length_scale):
    idx = torch.arange(n_sites)
    d = (idx[:, None] - idx[None, :]).abs()
    d = torch.minimum(d, n_sites - d).float()  # circular distance
    return torch.exp(-0.5 * (d / length_scale) ** 2)


def _correlated(chol, n, members):
    # (n, n_sites, members) ~ N(0, Sigma), iid over n and members
    n_sites = chol.shape[0]
    z = torch.randn(n, n_sites, members)
    return torch.einsum("ij,njm->nim", chol, z)


def _collection_coverage(ensemble_btscm, y_true_btsc, level=0.90):
    region_members = ensemble_btscm.sum(dim=2)  # sum over spatial -> (n,1,1,M)
    region_true = y_true_btsc.sum(dim=2)  # (n,1,1)
    lo = torch.quantile(region_members, (1 - level) / 2, dim=-1)
    hi = torch.quantile(region_members, (1 + level) / 2, dim=-1)
    inside = (region_true >= lo) & (region_true <= hi)
    return inside.float().mean().item()


def _to_btsc(t):
    return t[:, None, :, None]  # (n, S) -> (n, 1, S, 1)


def _to_btscm(t):
    return t[:, None, :, None, :]  # (n, S, M) -> (n, 1, S, 1, M)


def test_emos_ecc_restores_collection_coverage_only_with_a_correct_template():
    torch.manual_seed(0)
    n_sites, length_scale, members = 16, 2.0, 64
    n_cal = n_test = 3000

    sigma = _ring_cov(n_sites, length_scale) + 1e-3 * torch.eye(n_sites)
    chol = torch.linalg.cholesky(sigma)
    per_site_std = torch.sqrt(torch.diag(sigma))  # 1.0 for this stationary kernel

    def truth(n):
        return _correlated(chol, n, 1)[..., 0]  # (n, S)

    def diagonal_model(n):  # correct per-site marginal, spatially white
        return torch.randn(n, n_sites, members) * per_site_std[None, :, None]

    y_cal, y_test = truth(n_cal), truth(n_test)
    yp_cal, yp_test = diagonal_model(n_cal), diagonal_model(n_test)

    composite = EMOSECC(spatial_dims=(n_sites,))
    composite.calibrate(_to_btsc(y_cal), _to_btscm(yp_cal))

    yp_test_b = _to_btscm(yp_test)
    yt_test_b = _to_btsc(y_test)

    # EMOS alone fixes the per-site marginal (predict is reorder-invariant)
    intervals = composite.predict(yp_test_b, 0.10)
    lower, upper = intervals[..., 0, 0], intervals[..., 1, 0]
    per_site_cov = ((yt_test_b >= lower) & (yt_test_b <= upper)).float().mean().item()
    assert 0.86 < per_site_cov < 0.94

    correct_template = _to_btscm(_correlated(chol, n_test, members))
    wrong_template = _to_btscm(diagonal_model(n_test))  # spatially white
    oracle = _to_btscm(_correlated(chol, n_test, members))

    cov_correct = _collection_coverage(
        composite.sample(yp_test_b, members, template=correct_template), yt_test_b
    )
    cov_wrong = _collection_coverage(
        composite.sample(yp_test_b, members, template=wrong_template), yt_test_b
    )
    cov_oracle = _collection_coverage(oracle, yt_test_b)

    # a correct template recovers regional coverage to ~oracle; a wrong one does not
    assert cov_wrong < 0.65
    assert cov_correct > cov_wrong + 0.15
    assert abs(cov_correct - cov_oracle) < 0.12
