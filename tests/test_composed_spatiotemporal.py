"""Acceptance test: EMOS + ECC on a jointly spatio-temporal forecast.

The realistic both-at-once case. The truth carries temporal (AR over leads) AND
spatial (ring over sites) dependence simultaneously, via a *separable* covariance
-- the Kronecker product of the two kernels -- over the flattened ``(lead, site)``
positions. A model that is white in both has the correct per-``(lead, site)``
marginal but neither dependence, so both the spatial total (sum over sites) and
the temporal total (sum over leads) are under-covered. With a single template
carrying both correlations, the one EMOS + ECC pipeline -- over one
``(B, T, S, C, M)`` tensor, no axis switch -- restores both aggregates to the
oracle at once.
"""

import torch

from autouq.calibrators import EMOSECC

from toy_problems import (
    ar_cov,
    as_ensemble,
    as_truth,
    collection_coverage,
    correlated_normal,
    psd_correlation,
    ring_cov,
)

# canonical layout (B, T, S, C, M): both dependences are present, so both the
# temporal total (sum over leads) and the spatial total (sum over sites) matter
TIME_AXIS, SPACE_AXIS = 1, 2


def test_emos_ecc_restores_both_spatial_and_temporal_coverage_jointly():
    torch.manual_seed(0)
    n_leads, n_sites, members = 4, 8, 64
    n_cal = n_test = 3000

    # separable joint covariance over flattened (lead, site) positions; the
    # Kronecker product inherits the ring kernel's non-PD-ness, so project it
    # back onto the correlation cone before factoring
    sigma_t = ar_cov(n_leads, 0.7)
    sigma_s = ring_cov(n_sites, 1.5)
    sigma = psd_correlation(torch.kron(sigma_t, sigma_s))
    chol = torch.linalg.cholesky(sigma)

    def correlated_grid(n, members):
        # draw over the flat (lead, site) index, then unpack to the (T, S) grid
        flat = correlated_normal(chol, n, members)  # (B, T*S, M)
        return flat.reshape(n, n_leads, n_sites, members)  # (B, T, S, M)

    def truth(n):
        return correlated_grid(n, 1)[..., 0]  # (B, T, S)

    def white_model(n):
        # correct unit per-(lead, site) marginal, no dependence of either kind
        return torch.randn(n, n_leads, n_sites, members)  # (B, T, S, M)

    y_cal, y_test = truth(n_cal), truth(n_test)
    yp_cal, yp_test = white_model(n_cal), white_model(n_test)

    composite = EMOSECC()
    composite.calibrate(as_truth(y_cal), as_ensemble(yp_cal))

    yp_b, yt_b = as_ensemble(yp_test), as_truth(y_test)
    correct_template = as_ensemble(correlated_grid(n_test, members))
    oracle = as_ensemble(correlated_grid(n_test, members))

    ens_correct = composite.sample(yp_b, members, template=correct_template)
    ens_white = composite.sample(yp_b, members)  # template defaults to y_pred (white)

    # the correct template restores BOTH the temporal total (sum over leads) and
    # the spatial total (sum over sites) to ~oracle at once
    for axis in (TIME_AXIS, SPACE_AXIS):
        cov_white = collection_coverage(ens_white, yt_b, axis)
        cov_correct = collection_coverage(ens_correct, yt_b, axis)
        cov_oracle = collection_coverage(oracle, yt_b, axis)
        assert cov_white < 0.80
        assert cov_correct > cov_white + 0.10
        assert abs(cov_correct - cov_oracle) < 0.10
