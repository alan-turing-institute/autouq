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
    assert emos._beta1 is not None  # set by calibrate
    assert 1.8 < float(emos._beta1) < 2.2

    # the calibrated spread tracks the true residual scale (~1)
    _, sigma_cal = emos._calibrated_mean_std(yp_b)
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
    with pytest.raises(ValueError, match="match the fit"):
        emos.predict(torch.randn(200, 1, 4, 1, 16), 0.10)


def test_predict_rejects_invalid_alpha():
    emos = EMOS(spatial_dims=(1,))
    emos.calibrate(torch.randn(50, 1, 1, 1), torch.randn(50, 1, 1, 1, 8))
    with pytest.raises(ValueError, match="open interval"):
        emos.predict(torch.randn(50, 1, 1, 1, 8), 0.0)


def test_sample_is_reproducible_with_generator():
    torch.manual_seed(0)
    n, n_sites, members = 50, 3, 16
    y_b = torch.randn(n, 1, n_sites, 1)
    yp_b = torch.randn(n, 1, n_sites, 1, members)
    emos = EMOS(spatial_dims=(n_sites,))
    emos.calibrate(y_b, yp_b)

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    assert torch.equal(
        emos.sample(yp_b, members, generator=g1),
        emos.sample(yp_b, members, generator=g2),
    )
    g3 = torch.Generator().manual_seed(456)
    assert not torch.equal(
        emos.sample(yp_b, members, generator=g1),
        emos.sample(yp_b, members, generator=g3),
    )


def test_emos_fits_per_lead_temporal():
    # lead-varying gain: per-lead (default) coefficients must differ by lead and
    # bring every lead to nominal coverage
    torch.manual_seed(0)
    n, n_leads, members = 3000, 4, 30
    gains = torch.tensor([0.3, 0.6, 1.2, 2.0])
    x = torch.randn(n, n_leads)
    y = x + torch.randn(n, n_leads)  # truth: slope 1 in x
    members_t = gains[None, :, None] * x[:, :, None] + 0.3 * torch.randn(
        n, n_leads, members
    )
    y_b = y[:, :, None, None]  # (n, T, 1, 1)
    yp_b = members_t[:, :, None, None, :]  # (n, T, 1, 1, M)

    emos = EMOS(spatial_dims=(1,))  # default per=("time",)
    emos.calibrate(y_b, yp_b)

    intervals = emos.predict(yp_b, 0.10)
    lower, upper = intervals[..., 0, 0], intervals[..., 1, 0]
    per_lead_cov = ((y_b >= lower) & (y_b <= upper)).float().mean(dim=(0, 2, 3))
    assert torch.all((per_lead_cov > 0.86) & (per_lead_cov < 0.94))

    assert emos._beta1 is not None
    assert emos._beta1.numel() == n_leads  # one coefficient per lead
    assert emos._beta1.std() > 0.3  # genuinely lead-specific (~1/gains)


def test_emos_fits_per_site_spatial():
    # the spatial twin: site-varying gain, no time axis. per=("space",) fits
    # per-site coefficients and corrects each site individually
    torch.manual_seed(0)
    n, n_sites, members = 3000, 4, 30
    gains = torch.tensor([0.3, 0.6, 1.2, 2.0])
    x = torch.randn(n, n_sites)
    y = x + torch.randn(n, n_sites)
    members_t = gains[None, :, None] * x[:, :, None] + 0.3 * torch.randn(
        n, n_sites, members
    )
    y_b = y[:, None, :, None]  # (n, 1, S, 1)
    yp_b = members_t[:, None, :, None, :]  # (n, 1, S, 1, M)

    emos = EMOS(spatial_dims=(n_sites,), per=("space",))
    emos.calibrate(y_b, yp_b)

    intervals = emos.predict(yp_b, 0.10)
    lower, upper = intervals[..., 0, 0], intervals[..., 1, 0]
    per_site_cov = ((y_b >= lower) & (y_b <= upper)).float().mean(dim=(0, 1, 3))
    assert torch.all((per_site_cov > 0.86) & (per_site_cov < 0.94))

    assert emos._beta1 is not None
    assert emos._beta1.numel() == n_sites  # one coefficient per site
    assert emos._beta1.std() > 0.3  # genuinely site-specific (~1/gains)


def test_emos_recovers_under_large_scale():
    # standardisation makes the fit scale-invariant: same recovery at 1e6 scale
    torch.manual_seed(0)
    n, n_leads, members = 3000, 3, 30
    scale = 1e6
    x = torch.randn(n, n_leads)
    y = (x + torch.randn(n, n_leads)) * scale  # truth: y = a*x + noise, a = 1
    members_t = (0.5 * x[:, :, None] + 0.3 * torch.randn(n, n_leads, members)) * scale
    y_b = y[:, :, None, None]
    yp_b = members_t[:, :, None, None, :]

    emos = EMOS(spatial_dims=(1,))
    emos.calibrate(y_b, yp_b)

    cov = _per_alpha_coverage(emos.predict(yp_b, 0.10), y_b)
    assert 0.87 < cov[0].item() < 0.93
    assert emos._beta1 is not None
    assert torch.all((emos._beta1 > 1.8) & (emos._beta1 < 2.2))  # gain ~ a/b = 2
