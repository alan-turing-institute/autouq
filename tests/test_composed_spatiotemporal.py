"""Acceptance test: EMOS + ECC on a jointly spatio-temporal forecast.

The realistic both-at-once case. The truth carries temporal (AR over leads) AND
spatial (ring over sites) dependence at the same time, via a separable
covariance over the flattened (lead, site) positions. A model that is white in
both has the correct per-(lead, site) marginal but neither dependence, so both
the spatial total (sum over sites) and the temporal total (sum over leads) are
under-covered. With a single template carrying both correlations, the one
EMOS + ECC pipeline -- over one ``(B, T, S, C, M)`` tensor, no axis switch --
restores both aggregates to the oracle.
"""

import torch

from autouq.calibrators.base import ComposedCalibrator
from autouq.calibrators.ecc import ECC
from autouq.calibrators.emos import EMOS


def _ar_cov(n, rho):
    i = torch.arange(n)
    return rho ** (i[:, None] - i[None, :]).abs().float()


def _ring_cov(n, length_scale):
    i = torch.arange(n)
    d = (i[:, None] - i[None, :]).abs()
    d = torch.minimum(d, n - d).float()  # circular distance
    return torch.exp(-0.5 * (d / length_scale) ** 2)


def _psd_correlation(sigma):
    # the circular-distance ring kernel is not guaranteed positive-definite, and
    # the Kronecker product inherits that; clip to the PSD cone, then rescale to
    # a unit-diagonal correlation matrix
    w, v = torch.linalg.eigh(sigma)
    sigma = (v * w.clamp(min=1e-4)) @ v.T
    d = torch.sqrt(torch.diag(sigma))
    return sigma / torch.outer(d, d)


def _correlated(chol, n, n_leads, n_sites, members):
    # (n, T, S, members) ~ N(0, Sigma) over the flattened (lead, site) positions
    z = torch.randn(n, n_leads * n_sites, members)
    f = torch.einsum("ij,njm->nim", chol, z)
    return f.reshape(n, n_leads, n_sites, members)


def _collection_coverage(ensemble_btscm, y_true_btsc, axis, level=0.90):
    region_members = ensemble_btscm.sum(dim=axis)  # aggregate over leads or sites
    region_true = y_true_btsc.sum(dim=axis)
    lo = torch.quantile(region_members, (1 - level) / 2, dim=-1)
    hi = torch.quantile(region_members, (1 + level) / 2, dim=-1)
    inside = (region_true >= lo) & (region_true <= hi)
    return inside.float().mean().item()


def _to_btsc(t):
    return t[:, :, :, None]  # (n, T, S) -> (n, T, S, 1)


def _to_btscm(t):
    return t[:, :, :, None, :]  # (n, T, S, M) -> (n, T, S, 1, M)


def test_emos_ecc_restores_both_spatial_and_temporal_coverage_jointly():
    torch.manual_seed(0)
    n_leads, n_sites, members = 4, 8, 64
    n_cal = n_test = 3000

    sigma_t = _ar_cov(n_leads, 0.7)
    sigma_s = _ring_cov(n_sites, 1.5)
    sigma = _psd_correlation(torch.kron(sigma_t, sigma_s))
    chol = torch.linalg.cholesky(sigma)

    def truth(n):
        return _correlated(chol, n, n_leads, n_sites, 1)[..., 0]  # (n, T, S)

    def white_model(n):  # correct unit per-(lead, site) marginal, no dependence
        return torch.randn(n, n_leads, n_sites, members)

    y_cal, y_test = truth(n_cal), truth(n_test)
    yp_cal, yp_test = white_model(n_cal), white_model(n_test)

    composite = ComposedCalibrator(EMOS(spatial_dims=(n_sites,)), ECC())
    composite.calibrate(_to_btsc(y_cal), _to_btscm(yp_cal))

    yp_b, yt_b = _to_btscm(yp_test), _to_btsc(y_test)
    correct_template = _to_btscm(_correlated(chol, n_test, n_leads, n_sites, members))
    oracle = _to_btscm(_correlated(chol, n_test, n_leads, n_sites, members))

    ens_correct = composite.sample(yp_b, members, template=correct_template)
    ens_white = composite.sample(yp_b, members)  # template defaults to y_pred (white)

    # spatial aggregate = sum over sites (axis 2); temporal aggregate = sum over
    # leads (axis 1); the correct template restores BOTH to ~oracle at once
    for axis in (1, 2):
        cov_white = _collection_coverage(ens_white, yt_b, axis)
        cov_correct = _collection_coverage(ens_correct, yt_b, axis)
        cov_oracle = _collection_coverage(oracle, yt_b, axis)
        assert cov_white < 0.80
        assert cov_correct > cov_white + 0.10
        assert abs(cov_correct - cov_oracle) < 0.10
