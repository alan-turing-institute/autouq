"""Acceptance test: EMOS + ECC on a temporally correlated forecast.

The *temporal* twin of ``test_composed_ring``. A model that is white in time has
the correct per-lead marginal but no lead-to-lead coherence, so the time
aggregate (the sum over leads -- an accumulated total over a rollout) is badly
under-covered. EMOS leaves the marginals (already correct) alone; ECC restores
the temporal coherence, but only when reordered onto a template that carries the
true lead-to-lead dependence. ECC has no axis-specific logic, so this is the
same code path as the spatial ring test, applied to the time axis.
"""

import torch

from autouq.calibrators import EMOSECC

from toy_problems import (
    ar_cov,
    as_ensemble,
    as_truth,
    collection_coverage,
    correlated_normal,
)

# canonical layout (B, T, S, C, M): this toy's dependence lives across leads, so
# the discriminating aggregate is the sum over the time axis
TIME_AXIS = 1


def test_emos_ecc_restores_temporal_coherence_only_with_a_correct_template():
    torch.manual_seed(0)
    n_leads, rho, members = 16, 0.7, 64
    n_cal = n_test = 3000

    sigma = ar_cov(n_leads, rho)  # AR(1): positive-definite, unit diagonal
    chol = torch.linalg.cholesky(sigma)
    per_lead_std = torch.sqrt(torch.diag(sigma))  # 1.0 for this AR(1) kernel

    def truth(n):
        # one temporally correlated trajectory per case, at a single site (S = 1)
        traj = correlated_normal(chol, n, members=1)[..., 0]  # (B, T)
        return traj[:, :, None]  # (B, T, S=1)

    def white_model(n):
        # correct per-lead marginal but white in time (independent members)
        white = torch.randn(n, n_leads, members) * per_lead_std[None, :, None]
        return white[:, :, None, :]  # (B, T, S=1, M)

    y_cal, y_test = truth(n_cal), truth(n_test)
    yp_cal, yp_test = white_model(n_cal), white_model(n_test)

    composite = EMOSECC()
    composite.calibrate(as_truth(y_cal), as_ensemble(yp_cal))

    yp_test_b, yt_test_b = as_ensemble(yp_test), as_truth(y_test)

    # EMOS alone fixes the per-lead marginal (predict is reorder-invariant)
    intervals = composite.predict(yp_test_b, 0.10)
    lower, upper = intervals[..., 0, 0], intervals[..., 1, 0]
    per_lead_cov = ((yt_test_b >= lower) & (yt_test_b <= upper)).float().mean().item()
    assert 0.86 < per_lead_cov < 0.94

    # a correct template carries the true AR(1) dependence; a white one does not;
    # the oracle is an independent draw of the true correlated ensemble
    correct_template = as_ensemble(correlated_normal(chol, n_test, members)[:, :, None])
    wrong_template = as_ensemble(white_model(n_test))
    oracle = as_ensemble(correlated_normal(chol, n_test, members)[:, :, None])

    cov_correct = collection_coverage(
        composite.sample(yp_test_b, members, template=correct_template),
        yt_test_b,
        TIME_AXIS,
    )
    cov_wrong = collection_coverage(
        composite.sample(yp_test_b, members, template=wrong_template),
        yt_test_b,
        TIME_AXIS,
    )
    cov_oracle = collection_coverage(oracle, yt_test_b, TIME_AXIS)

    # a correct template recovers temporal-aggregate coverage to ~oracle; a white
    # one leaves it badly under-covered (an independent ensemble cannot match the
    # variance of a positively autocorrelated sum)
    assert cov_wrong < 0.70
    assert cov_correct > cov_wrong + 0.15
    assert abs(cov_correct - cov_oracle) < 0.12
