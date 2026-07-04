"""Acceptance test: EMOS + ECC on a spatially correlated ring of sites.

The *spatial* instance of the shared story in ``toy_problems``. A "diagonal"
forecast head is spatially white -- correct per-site marginal, zero cross-site
correlation -- so the regional total (the sum over sites) is badly
under-covered. EMOS leaves the already-correct marginals alone; ECC restores the
joint spread, but only when the calibrated marginals are reordered onto a
template that carries the true ring dependence. A white template leaves the
regional total under-covered; a correctly correlated one recovers it to ~oracle.
"""

import torch

from autouq.calibrators import EMOSECC

from toy_problems import (
    as_ensemble,
    as_truth,
    collection_coverage,
    correlated_normal,
    ring_cov,
)

# canonical layout (B, T, S, C, M): this toy's dependence lives across sites, so
# the discriminating aggregate is the sum over the spatial axis
SPACE_AXIS = 2


def test_emos_ecc_restores_collection_coverage_only_with_a_correct_template():
    torch.manual_seed(0)
    n_sites, length_scale, members = 16, 2.0, 64
    n_cal = n_test = 3000

    # the ring kernel is only positive-semi-definite on the wrap-around metric;
    # a tiny diagonal jitter lifts it to strictly PD for the Cholesky factor
    sigma = ring_cov(n_sites, length_scale) + 1e-3 * torch.eye(n_sites)
    chol = torch.linalg.cholesky(sigma)
    per_site_std = torch.sqrt(torch.diag(sigma))  # ~1 for this stationary kernel

    def truth(n):
        # one spatially correlated field per case, on a single lead (T = 1)
        field = correlated_normal(chol, n, members=1)[..., 0]  # (B, S)
        return field[:, None, :]  # (B, T=1, S)

    def diagonal_model(n):
        # correct per-site marginal but spatially white (independent members)
        white = torch.randn(n, n_sites, members) * per_site_std[None, :, None]
        return white[:, None, :, :]  # (B, T=1, S, M)

    y_cal, y_test = truth(n_cal), truth(n_test)
    yp_cal, yp_test = diagonal_model(n_cal), diagonal_model(n_test)

    composite = EMOSECC()
    composite.calibrate(as_truth(y_cal), as_ensemble(yp_cal))

    yp_test_b, yt_test_b = as_ensemble(yp_test), as_truth(y_test)

    # EMOS alone fixes the per-site marginal (predict is reorder-invariant)
    intervals = composite.predict(yp_test_b, 0.10)
    lower, upper = intervals[..., 0, 0], intervals[..., 1, 0]
    per_site_cov = ((yt_test_b >= lower) & (yt_test_b <= upper)).float().mean().item()
    assert 0.86 < per_site_cov < 0.94

    # a correct template carries the true ring dependence; a white one does not;
    # the oracle is an independent draw of the true correlated ensemble
    correct_template = as_ensemble(correlated_normal(chol, n_test, members)[:, None])
    wrong_template = as_ensemble(diagonal_model(n_test))
    oracle = as_ensemble(correlated_normal(chol, n_test, members)[:, None])

    cov_correct = collection_coverage(
        composite.sample(yp_test_b, members, template=correct_template),
        yt_test_b,
        SPACE_AXIS,
    )
    cov_wrong = collection_coverage(
        composite.sample(yp_test_b, members, template=wrong_template),
        yt_test_b,
        SPACE_AXIS,
    )
    cov_oracle = collection_coverage(oracle, yt_test_b, SPACE_AXIS)

    # a correct template recovers regional coverage to ~oracle; a wrong one does not
    assert cov_wrong < 0.65
    assert cov_correct > cov_wrong + 0.15
    assert abs(cov_correct - cov_oracle) < 0.12
