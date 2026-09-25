import math

import pytest
import torch

from autouq.calibrators.emos import EMOS, _GroupCRPS
from autouq.calibrators.grouping import AxisRole


def _per_alpha_coverage(intervals, y_true):
    # intervals (..., 2, A); y_true (...)
    lower = intervals[..., 0, :]
    upper = intervals[..., 1, :]
    y = y_true.unsqueeze(-1)
    inside = (y >= lower) & (y <= upper)
    dims = tuple(range(inside.ndim - 1))
    return inside.float().mean(dim=dims)


def _leads_that_differ(n_leads, n_samples, *, seed, noise_seed):
    """Truth from a known per-lead EMOS whose coefficients differ widely by lead.

    Per lead: ensemble mean ``m ~ N(0, amp^2)``, 10 members ``m + spread * noise``,
    truth ``~ N(b0 + b1 * xbar, g0 + g1 * s2)``. Amplitudes span 1e-2 to 10,
    spread/amplitude 1e-2 to 1, ``g1`` 0.1 to 30, and ``g0`` is 0 on about half
    the leads. Coefficients come from ``seed``, samples from ``noise_seed``, so
    one ``seed`` with two noise seeds gives a calibration and a test set.
    """
    coef = torch.Generator().manual_seed(seed)

    def log_uniform(low, high):
        exponent = torch.empty(n_leads).uniform_(
            math.log10(low), math.log10(high), generator=coef
        )
        return 10**exponent

    amp = log_uniform(1e-2, 10.0)
    spread = log_uniform(1e-2, 1.0) * amp
    b0 = torch.empty(n_leads).uniform_(-0.5, 0.5, generator=coef) * amp
    b1 = torch.empty(n_leads).uniform_(0.5, 1.5, generator=coef)
    g1 = log_uniform(0.1, 30.0)
    g0_nonzero = (log_uniform(1e-2, 0.3) * amp) ** 2
    g0 = torch.where(torch.rand(n_leads, generator=coef) < 0.5, 0.0, g0_nonzero)

    noise = torch.Generator().manual_seed(noise_seed)
    m = torch.randn(n_samples, n_leads, generator=noise) * amp
    members = m[..., None] + spread[:, None] * torch.randn(
        n_samples, n_leads, 10, generator=noise
    )
    xbar, s2 = members.mean(dim=-1), members.var(dim=-1)
    y = (
        b0
        + b1 * xbar
        + (g0 + g1 * s2).sqrt() * torch.randn(n_samples, n_leads, generator=noise)
    )
    return y[:, :, None, None], members[:, :, None, None, :]  # (B, T, 1, 1[, M])


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


def test_emos_reaches_nominal_coverage_on_every_lead_when_leads_differ():
    # 100 leads whose true coefficients differ widely: fitting them in one pooled
    # optimisation stalled short of most leads' optimum, leaving about half of
    # them far from nominal coverage
    y, pred = _leads_that_differ(100, 2048, seed=0, noise_seed=1)
    y_test, pred_test = _leads_that_differ(100, 2048, seed=0, noise_seed=2)

    emos = EMOS()
    emos.calibrate(y, pred)

    intervals = emos.predict(pred_test, 0.10)
    lower, upper = intervals[..., 0, 0], intervals[..., 1, 0]
    inside = (y_test >= lower) & (y_test <= upper)
    per_lead_cov = inside.float().mean(dim=(0, 2, 3))
    assert torch.all((per_lead_cov > 0.85) & (per_lead_cov < 0.95))


def test_emos_fit_over_all_leads_matches_fitting_each_lead_alone():
    # each lead is an independent problem, so fitting all leads at once must give
    # the same predictive as fitting every lead on its own
    n_leads = 12
    y, pred = _leads_that_differ(n_leads, 2048, seed=3, noise_seed=4)
    joint = EMOS()
    joint.calibrate(y, pred)
    mu, sigma = joint._calibrated_mean_std(pred)

    for t in range(n_leads):
        alone = EMOS()
        alone.calibrate(y[:, t : t + 1], pred[:, t : t + 1])
        mu_t, sigma_t = alone._calibrated_mean_std(pred[:, t : t + 1])
        scale = y[:, t].std().item()
        torch.testing.assert_close(mu[:, t : t + 1], mu_t, rtol=0, atol=1e-4 * scale)
        torch.testing.assert_close(sigma[:, t : t + 1], sigma_t, rtol=1e-4, atol=0)


def test_calibrate_keeps_a_constant_truth_group_sharp():
    # a masked or dead cell carries no signal: its fitted predictive must stay
    # sharp, not be inflated by a Newton step the conditioning crushed to zero
    torch.manual_seed(0)
    y = torch.randn(256, 1, 4, 1)
    y[:, :, 2:] = 0.0
    pred = y.unsqueeze(-1) + 0.5 * torch.randn(256, 1, 4, 1, 8)

    emos = EMOS(per=(AxisRole.SPACE,))
    emos.calibrate(y, pred)
    _, sigma = emos._calibrated_mean_std(pred)

    assert sigma[:, :, 2:].max() < 1e-4  # the constant cells
    assert sigma[:, :, :2].min() > 1e-2  # the live ones are untouched


def test_calibrate_does_not_amplify_a_collapsed_ensemble_group():
    # every member equal at one lead leaves gamma1 unidentifiable there; it must
    # not come back inflated when predicting on an ensemble that does spread
    torch.manual_seed(0)
    y = torch.randn(256, 2, 3, 1)
    pred = y.unsqueeze(-1) + 0.5 * torch.randn(256, 2, 3, 1, 8)
    pred[:, 1] = pred[:, 1, ..., :1].expand_as(pred[:, 1])

    emos = EMOS()
    emos.calibrate(y, pred)
    spreading = y.unsqueeze(-1) + 0.5 * torch.randn(256, 2, 3, 1, 8)
    _, sigma = emos._calibrated_mean_std(spreading)

    assert sigma[:, 1].max() < 10 * sigma[:, 0].max()


def test_calibrate_rejects_non_finite_inputs():
    torch.manual_seed(0)
    y = torch.randn(64, 2, 3, 1)
    pred = y.unsqueeze(-1) + torch.randn(64, 2, 3, 1, 8)
    pred[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        EMOS().calibrate(y, pred)


def test_calibrate_warns_when_groups_do_not_converge():
    y, pred = _leads_that_differ(4, 512, seed=5, noise_seed=6)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        EMOS(max_iter=1).calibrate(y, pred)


def test_predict_and_sample_keep_the_forecast_dtype():
    # coefficients are fitted in float64, but outputs follow the forecast's dtype
    torch.manual_seed(0)
    y_b = torch.randn(200, 2, 3, 1)
    yp_b = torch.randn(200, 2, 3, 1, 8)
    emos = EMOS()
    emos.calibrate(y_b, yp_b)

    assert emos.predict(yp_b, 0.10).dtype == torch.float32
    assert emos.sample(yp_b, 4).dtype == torch.float32


def test_group_crps_derivatives_match_autograd():
    # the closed-form gradient and Hessian that drive the fit, against autograd;
    # groups are independent, so the Hessian of the sum is block-diagonal
    torch.manual_seed(0)
    n_groups = 3
    y = torch.randn(50, n_groups, 4, 2, dtype=torch.float64)
    xbar = y + 0.5 * torch.randn_like(y)
    s2 = torch.rand_like(y) + 0.1
    objective = _GroupCRPS(y, xbar, s2, view=(1, -1, 1, 1), pooled=(0, 2, 3))
    theta = torch.tensor(
        [[0.1, -0.2, 0.3], [0.9, 1.1, 0.7], [0.2, 0.05, 0.4], [0.5, 1.5, 0.8]],
        dtype=torch.float64,
    )

    losses, grad, hess = objective.derivatives(theta)

    def total(t):
        return objective(t).sum()

    torch.testing.assert_close(losses, objective(theta))
    torch.testing.assert_close(grad, torch.autograd.functional.jacobian(total, theta))
    full = torch.autograd.functional.hessian(total, theta)  # (4, G, 4, G)
    blocks = torch.stack([full[:, g, :, g] for g in range(n_groups)], dim=-1)
    torch.testing.assert_close(hess, blocks)
