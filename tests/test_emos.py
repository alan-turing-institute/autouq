import pytest
import torch

from autouq.calibrators.emos import EMOS


def _per_alpha_coverage(intervals, y_true):
    # intervals (..., 2, A); y_true (...)
    lower = intervals[..., 0, :]
    upper = intervals[..., 1, :]
    y = y_true.unsqueeze(-1)
    inside = (y >= lower) & (y <= upper)
    dims = tuple(range(inside.ndim - 1))
    return inside.float().mean(dim=dims)


def _corr(a, b):
    a = a.reshape(-1) - a.mean()
    b = b.reshape(-1) - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm())).item()


def test_emos_recovers_gain_and_nominal_coverage_from_underdispersed_head():
    torch.manual_seed(0)
    n, members = 3000, 30
    a, sigma_true = 1.0, 1.0  # truth: y = a*x + N(0, sigma_true)
    b, sigma_model = 0.5, 0.3  # model: under-dispersed, wrong gain

    x = torch.randn(n)
    y = a * x + sigma_true * torch.randn(n)
    members_t = b * x[:, None] + sigma_model * torch.randn(n, members)

    y_b = y[:, None, None, None]  # (n, T=1, S=1, C=1)
    yp_b = members_t[:, None, None, None, :]  # (n, 1, 1, 1, M)

    # raw (uncalibrated) head badly under-covers
    raw_mu = members_t.mean(dim=-1)
    raw_sigma = members_t.std(dim=-1)
    z90 = 1.6448536
    raw_inside = (y >= raw_mu - z90 * raw_sigma) & (y <= raw_mu + z90 * raw_sigma)
    assert raw_inside.float().mean().item() < 0.6

    emos = EMOS(spatial_dims=(1,))
    emos.calibrate(y_b, yp_b)

    cov = _per_alpha_coverage(emos.predict(yp_b, 0.10), y_b)
    assert 0.87 < cov[0].item() < 0.93

    # the wrong gain b=0.5 is corrected to beta1 ~ a/b = 2
    assert 1.8 < float(emos._beta1) < 2.2

    # the calibrated spread tracks the true residual scale (~1)
    _, sigma_cal = emos._calibrated_moments(yp_b)
    assert 0.9 < sigma_cal.mean().item() < 1.15


def test_predict_and_sample_shapes():
    torch.manual_seed(0)
    n, n_sites, members = 200, 4, 16
    y_b = torch.randn(n, 1, n_sites, 1)
    yp_b = torch.randn(n, 1, n_sites, 1, members)

    emos = EMOS(spatial_dims=(n_sites,))
    emos.calibrate(y_b, yp_b)

    intervals = emos.predict(yp_b, [0.1, 0.05])
    assert intervals.shape == (n, 1, n_sites, 1, 2, 2)

    samples = emos.sample(yp_b, 50)
    assert samples.shape == (n, 1, n_sites, 1, 50)


def test_sample_is_cross_site_independent():
    torch.manual_seed(0)
    n, n_sites, members = 1000, 4, 200
    y_b = torch.randn(n, 1, n_sites, 1)
    yp_b = torch.randn(n, 1, n_sites, 1, members)

    emos = EMOS(spatial_dims=(n_sites,))
    emos.calibrate(y_b, yp_b)
    s = emos.sample(yp_b, members).squeeze()  # (n, n_sites, members)

    assert abs(_corr(s[:, 0, :], s[:, 1, :])) < 0.1


def test_predict_before_calibrate_raises():
    emos = EMOS(spatial_dims=(1,))
    with pytest.raises(RuntimeError, match="calibrate"):
        emos.predict(torch.randn(2, 1, 1, 1, 5), 0.1)


def test_calibrate_rejects_single_member_ensemble():
    emos = EMOS(spatial_dims=(1,))
    with pytest.raises(ValueError, match="at least 2 members"):
        emos.calibrate(torch.randn(50, 1, 1, 1), torch.randn(50, 1, 1, 1, 1))


def test_predict_rejects_lead_count_mismatch():
    # fit on 3 leads, then predict a single-lead tensor: must error, not
    # silently broadcast the fitted coefficients onto the size-1 lead axis
    emos = EMOS(spatial_dims=(4,))
    emos.calibrate(torch.randn(200, 3, 4, 1), torch.randn(200, 3, 4, 1, 16))
    with pytest.raises(ValueError, match="lead"):
        emos.predict(torch.randn(200, 1, 4, 1, 16), 0.10)
