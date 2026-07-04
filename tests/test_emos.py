import pytest
import torch

from autouq.calibrators.emos import EMOS
from autouq.calibrators.grouping import AxisRole


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

    emos = EMOS()
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

    emos = EMOS()
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

    emos = EMOS()
    emos.calibrate(y_b, yp_b)
    s = emos.sample(yp_b, members).squeeze()  # (n, n_sites, members)

    assert abs(_corr(s[:, 0, :], s[:, 1, :])) < 0.1


def test_predict_before_calibrate_raises():
    emos = EMOS()
    with pytest.raises(RuntimeError, match="calibrate"):
        emos.predict(torch.randn(2, 1, 1, 1, 5), 0.1)


def test_calibrate_rejects_single_member_ensemble():
    emos = EMOS()
    with pytest.raises(ValueError, match="at least 2 members"):
        emos.calibrate(torch.randn(50, 1, 1, 1), torch.randn(50, 1, 1, 1, 1))


def test_predict_rejects_lead_count_mismatch():
    # fit on 3 leads, then predict a single-lead tensor: must error, not
    # silently broadcast the fitted coefficients onto the size-1 lead axis
    emos = EMOS()
    emos.calibrate(torch.randn(200, 3, 4, 1), torch.randn(200, 3, 4, 1, 16))
    with pytest.raises(ValueError, match="match the fit"):
        emos.predict(torch.randn(200, 1, 4, 1, 16), 0.10)


def test_predict_rejects_invalid_alpha():
    emos = EMOS()
    emos.calibrate(torch.randn(50, 1, 1, 1), torch.randn(50, 1, 1, 1, 8))
    with pytest.raises(ValueError, match="open interval"):
        emos.predict(torch.randn(50, 1, 1, 1, 8), 0.0)


def test_sample_is_reproducible_with_generator():
    torch.manual_seed(0)
    n, n_sites, members = 50, 3, 16
    y_b = torch.randn(n, 1, n_sites, 1)
    yp_b = torch.randn(n, 1, n_sites, 1, members)
    emos = EMOS()
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

    emos = EMOS()  # default per=(AxisRole.TIME,)
    emos.calibrate(y_b, yp_b)

    intervals = emos.predict(yp_b, 0.10)
    lower, upper = intervals[..., 0, 0], intervals[..., 1, 0]
    per_lead_cov = ((y_b >= lower) & (y_b <= upper)).float().mean(dim=(0, 2, 3))
    assert torch.all((per_lead_cov > 0.86) & (per_lead_cov < 0.94))

    assert emos._beta1 is not None
    assert emos._beta1.numel() == n_leads  # one coefficient per lead
    assert emos._beta1.std() > 0.3  # genuinely lead-specific (~1/gains)


def test_emos_fits_per_site_spatial():
    # the spatial twin: site-varying gain, no time axis. per=(AxisRole.SPACE,) fits
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

    emos = EMOS(per=(AxisRole.SPACE,))
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

    emos = EMOS()
    emos.calibrate(y_b, yp_b)

    cov = _per_alpha_coverage(emos.predict(yp_b, 0.10), y_b)
    assert 0.87 < cov[0].item() < 0.93
    assert emos._beta1 is not None
    assert torch.all((emos._beta1 > 1.8) & (emos._beta1 < 2.2))  # gain ~ a/b = 2


def test_emos_normalizes_per_to_enum_members():
    # strings are accepted at the boundary but stored as AxisRole members, so
    # internal state is never stringly-typed
    assert EMOS(per=("time", "space")).per == (AxisRole.TIME, AxisRole.SPACE)
    assert EMOS(per=(AxisRole.CHANNEL,)).per == (AxisRole.CHANNEL,)
    assert all(isinstance(role, AxisRole) for role in EMOS().per)


def test_emos_fits_per_time_and_space_combination():
    # per=(TIME, SPACE) gets its own coefficients for every (lead, site) cell
    torch.manual_seed(0)
    n, n_leads, n_sites, members = 3000, 3, 4, 30
    gains = torch.linspace(0.3, 2.0, n_leads * n_sites).reshape(n_leads, n_sites)
    x = torch.randn(n, n_leads, n_sites)
    y = x + torch.randn(n, n_leads, n_sites)
    members_t = gains[None, :, :, None] * x[..., None] + 0.3 * torch.randn(
        n, n_leads, n_sites, members
    )
    y_b = y[..., None]  # (n, T, S, 1)
    yp_b = members_t[:, :, :, None, :]  # (n, T, S, 1, M)

    emos = EMOS(per=(AxisRole.TIME, AxisRole.SPACE))
    emos.calibrate(y_b, yp_b)

    assert emos._beta1 is not None
    assert tuple(emos._beta1.shape) == (n_leads, n_sites)  # one coeff per (lead, site)
    cov = _per_alpha_coverage(emos.predict(yp_b, 0.10), y_b)
    assert 0.86 < cov[0].item() < 0.94


def test_emos_per_space_expands_to_all_spatial_axes():
    # SPACE covers *every* spatial axis: on (B, T, H, W, C) a per=(AxisRole.SPACE,) fit
    # yields one coefficient per (H, W) cell, pooling over batch and time
    torch.manual_seed(0)
    n, n_leads, h, w, members = 2000, 2, 3, 4, 30
    gains = torch.linspace(0.3, 2.0, h * w).reshape(h, w)
    x = torch.randn(n, n_leads, h, w)
    y = x + torch.randn(n, n_leads, h, w)
    members_t = gains[None, None, :, :, None] * x[..., None] + 0.3 * torch.randn(
        n, n_leads, h, w, members
    )
    y_b = y[..., None]  # (n, T, H, W, 1)
    yp_b = members_t[:, :, :, :, None, :]  # (n, T, H, W, 1, M)

    emos = EMOS(per=(AxisRole.SPACE,))
    emos.calibrate(y_b, yp_b)

    assert emos._beta1 is not None
    assert tuple(emos._beta1.shape) == (h, w)  # one coeff per spatial (H, W) cell
    cov = _per_alpha_coverage(emos.predict(yp_b, 0.10), y_b)
    assert 0.86 < cov[0].item() < 0.94


def test_emos_fits_per_channel():
    # per=(AxisRole.CHANNEL,) gets its own coefficients for each channel/variable
    torch.manual_seed(0)
    n, n_channels, members = 3000, 2, 30
    gains = torch.tensor([0.4, 1.8])
    x = torch.randn(n, n_channels)
    y = x + torch.randn(n, n_channels)
    members_t = gains[None, :, None] * x[:, :, None] + 0.3 * torch.randn(
        n, n_channels, members
    )
    y_b = y[:, None, None, :]  # (n, 1, 1, C)
    yp_b = members_t[:, None, None, :, :]  # (n, 1, 1, C, M)

    emos = EMOS(per=(AxisRole.CHANNEL,))
    emos.calibrate(y_b, yp_b)

    assert emos._beta1 is not None
    assert tuple(emos._beta1.shape) == (n_channels,)  # one coeff per channel
    assert emos._beta1.std() > 0.3  # genuinely channel-specific
    cov = _per_alpha_coverage(emos.predict(yp_b, 0.10), y_b)
    assert 0.86 < cov[0].item() < 0.94
