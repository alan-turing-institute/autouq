"""Acceptance test: EMOS + ECC on a temporally-correlated forecast.

The temporal twin of ``test_composed_ring``. A model that is white in time has
the correct per-lead marginal but no temporal coherence, so the time-aggregate
(the sum over leads, e.g. an accumulated total over a rollout) is badly
under-covered. EMOS leaves the marginals (already correct) alone; ECC restores
the temporal coherence -- but only when reordered onto a template that carries
the true lead-to-lead dependence. ECC has no axis-specific logic, so this is the
same code path as the spatial ring test, applied to the time axis.
"""

import torch

from autouq.calibrators.base import ComposedCalibrator
from autouq.calibrators.ecc import ECC
from autouq.calibrators.emos import EMOS


def _ar_cov(n_leads, rho):
    i = torch.arange(n_leads)
    return rho ** (i[:, None] - i[None, :]).abs().float()  # AR(1): unit diag, PD


def _correlated(chol, n, members):
    # (n, n_leads, members) ~ N(0, Sigma), iid over n and members
    n_leads = chol.shape[0]
    z = torch.randn(n, n_leads, members)
    return torch.einsum("ij,njm->nim", chol, z)


def _collection_coverage(ensemble_btscm, y_true_btsc, level=0.90):
    region_members = ensemble_btscm.sum(dim=1)  # sum over leads -> (n,1,1,M)
    region_true = y_true_btsc.sum(dim=1)  # (n,1,1)
    lo = torch.quantile(region_members, (1 - level) / 2, dim=-1)
    hi = torch.quantile(region_members, (1 + level) / 2, dim=-1)
    inside = (region_true >= lo) & (region_true <= hi)
    return inside.float().mean().item()


def _to_btsc(t):
    return t[:, :, None, None]  # (n, T) -> (n, T, 1, 1)


def _to_btscm(t):
    return t[:, :, None, None, :]  # (n, T, M) -> (n, T, 1, 1, M)


def test_emos_ecc_restores_temporal_coherence_only_with_a_correct_template():
    torch.manual_seed(0)
    n_leads, rho, members = 16, 0.7, 64
    n_cal = n_test = 3000

    sigma = _ar_cov(n_leads, rho)
    chol = torch.linalg.cholesky(sigma)
    per_lead_std = torch.sqrt(torch.diag(sigma))  # 1.0 for this AR(1) kernel

    def truth(n):
        return _correlated(chol, n, 1)[..., 0]  # (n, T)

    def white_model(n):  # correct per-lead marginal, white in time
        return torch.randn(n, n_leads, members) * per_lead_std[None, :, None]

    y_cal, y_test = truth(n_cal), truth(n_test)
    yp_cal, yp_test = white_model(n_cal), white_model(n_test)

    composite = ComposedCalibrator(EMOS(spatial_dims=(1,)), ECC())
    composite.calibrate(_to_btsc(y_cal), _to_btscm(yp_cal))

    yp_test_b = _to_btscm(yp_test)
    yt_test_b = _to_btsc(y_test)

    # EMOS alone fixes the per-lead marginal (predict is reorder-invariant)
    intervals = composite.predict(yp_test_b, 0.10)
    lower, upper = intervals[..., 0, 0], intervals[..., 1, 0]
    per_lead_cov = ((yt_test_b >= lower) & (yt_test_b <= upper)).float().mean().item()
    assert 0.86 < per_lead_cov < 0.94

    correct_template = _to_btscm(_correlated(chol, n_test, members))
    wrong_template = _to_btscm(white_model(n_test))  # white in time
    oracle = _to_btscm(_correlated(chol, n_test, members))

    cov_correct = _collection_coverage(
        composite.sample(yp_test_b, members, template=correct_template), yt_test_b
    )
    cov_wrong = _collection_coverage(
        composite.sample(yp_test_b, members, template=wrong_template), yt_test_b
    )
    cov_oracle = _collection_coverage(oracle, yt_test_b)

    # a correct template recovers temporal-aggregate coverage to ~oracle; a white
    # one leaves it badly under-covered (an independent ensemble cannot match the
    # variance of a positively autocorrelated sum)
    assert cov_wrong < 0.70
    assert cov_correct > cov_wrong + 0.15
    assert abs(cov_correct - cov_oracle) < 0.12
