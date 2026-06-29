"""EMOS / nonhomogeneous Gaussian regression marginal calibrator."""

import math
from collections.abc import Sequence

import torch
from torch.distributions import Normal
from torch.nn.functional import softplus

from autouq.calibrators.base import SamplableCalibrator
from autouq.types import Tensor, TensorBTSCM, TensorBTSIA

_STD_NORMAL = Normal(0.0, 1.0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_VAR_FLOOR = 1e-12


def _inv_softplus(x: float) -> float:
    """Inverse softplus, for initialising a softplus-constrained parameter."""
    return math.log(math.expm1(x))


def _std_normal_pdf(z: Tensor) -> Tensor:
    """Standard-normal density evaluated elementwise."""
    return torch.exp(-0.5 * z * z) / _SQRT_2PI


def _gaussian_crps(mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
    """Closed-form CRPS of a Gaussian predictive (Gneiting et al., 2005)."""
    z = (y - mu) / sigma
    cdf = _STD_NORMAL.cdf(z)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * _std_normal_pdf(z) - _INV_SQRT_PI)


class EMOS(SamplableCalibrator):
    r"""Ensemble Model Output Statistics (nonhomogeneous Gaussian regression).

    A parametric marginal calibrator. For each lead it fits an affine map from
    the raw ensemble's mean :math:`\bar{x}` and variance :math:`s^2` to a
    calibrated Gaussian predictive,

    .. math::
        \mu_{\mathrm{cal}} = \beta_0 + \beta_1 \bar{x}, \qquad
        \sigma^2_{\mathrm{cal}} = \gamma_0 + \gamma_1 s^2
        \quad (\gamma_0, \gamma_1 \ge 0),

    by minimising the mean closed-form Gaussian CRPS over the calibration set.
    The non-negativity of :math:`\gamma_0, \gamma_1` is enforced with a softplus
    link. It subsumes pure variance inflation as the constrained case
    :math:`\beta_0 = 0, \beta_1 = 1, \gamma_0 = 0`.

    Coefficients are fit per lead (the time axis), pooled over batch, space and
    channel. As a :class:`SamplableCalibrator` it can also draw an ensemble from
    the calibrated marginals (:meth:`sample`), for composition with a dependence
    calibrator.

    Parameters
    ----------
    spatial_dims
        Spatial dimension sizes, forwarded to :class:`Calibrator`.
    max_iter
        L-BFGS iterations for the per-lead CRPS fit.
    lr
        L-BFGS learning rate.
    """

    def __init__(
        self, spatial_dims: Sequence[int], *, max_iter: int = 100, lr: float = 1.0
    ):
        super().__init__(spatial_dims)
        self.max_iter = max_iter
        self.lr = lr
        self._beta0: Tensor | None = None
        self._beta1: Tensor | None = None
        self._raw_gamma0: Tensor | None = None
        self._raw_gamma1: Tensor | None = None

    @staticmethod
    def _moments(y_pred: Tensor) -> tuple[Tensor, Tensor]:
        """Raw ensemble mean and sample variance over the member axis."""
        n_members = y_pred.shape[-1]
        if n_members < 2:
            msg = (
                "EMOS needs an ensemble of at least 2 members to estimate "
                f"spread; got {n_members} on the last axis."
            )
            raise ValueError(msg)
        return y_pred.mean(dim=-1), y_pred.var(dim=-1, unbiased=True)

    @staticmethod
    def _lead_view(n_leads: int, ndim: int) -> tuple[int, ...]:
        """Broadcast shape placing per-lead coefficients on the time axis."""
        return (1, n_leads, *([1] * (ndim - 2)))

    def calibrate(self, y_true: Tensor, y_pred: Tensor):
        """Fit per-lead affine coefficients by minimising mean Gaussian CRPS."""
        xbar, s2 = self._moments(y_pred)
        n_leads = xbar.shape[1]
        view = self._lead_view(n_leads, xbar.ndim)
        opts = {"device": xbar.device, "dtype": xbar.dtype, "requires_grad": True}
        beta0 = torch.zeros(n_leads, **opts)
        beta1 = torch.ones(n_leads, **opts)
        raw_g0 = torch.full((n_leads,), _inv_softplus(1e-3), **opts)
        raw_g1 = torch.full((n_leads,), _inv_softplus(1.0), **opts)
        params = [beta0, beta1, raw_g0, raw_g1]
        optimizer = torch.optim.LBFGS(
            params, lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            mu = beta0.view(view) + beta1.view(view) * xbar
            sigma2 = softplus(raw_g0).view(view) + softplus(raw_g1).view(view) * s2
            sigma = torch.sqrt(sigma2 + _VAR_FLOOR)
            loss = _gaussian_crps(mu, sigma, y_true).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        self._beta0 = beta0.detach()
        self._beta1 = beta1.detach()
        self._raw_gamma0 = raw_g0.detach()
        self._raw_gamma1 = raw_g1.detach()

    def _calibrated_moments(self, y_pred: Tensor) -> tuple[Tensor, Tensor]:
        """Calibrated per-site Gaussian mean and standard deviation."""
        if (
            self._beta0 is None
            or self._beta1 is None
            or self._raw_gamma0 is None
            or self._raw_gamma1 is None
        ):
            msg = "EMOS.calibrate must be called before predict/sample."
            raise RuntimeError(msg)
        beta0, beta1 = self._beta0, self._beta1
        raw_g0, raw_g1 = self._raw_gamma0, self._raw_gamma1
        xbar, s2 = self._moments(y_pred)
        n_leads = beta0.shape[0]
        if xbar.shape[1] != n_leads:
            msg = (
                f"EMOS was fit for {n_leads} lead(s) but got {xbar.shape[1]} at "
                "predict/sample time; the lead (time) axis must match the fit."
            )
            raise ValueError(msg)
        view = self._lead_view(n_leads, xbar.ndim)
        mu = beta0.view(view) + beta1.view(view) * xbar
        sigma2 = softplus(raw_g0).view(view) + softplus(raw_g1).view(view) * s2
        return mu, torch.sqrt(sigma2 + _VAR_FLOOR)

    def predict(self, y_pred: Tensor, alphas: float | Sequence[float]) -> TensorBTSIA:
        """Central calibrated Gaussian intervals at each miscoverage ``alpha``."""
        alphas = [float(alphas)] if isinstance(alphas, int | float) else list(alphas)
        mu, sigma = self._calibrated_moments(y_pred)
        quantiles = torch.tensor(
            [1.0 - alpha / 2.0 for alpha in alphas], device=mu.device, dtype=mu.dtype
        )
        zs = _STD_NORMAL.icdf(quantiles)  # (n_alphas,)
        half_width = sigma[..., None] * zs  # (..., n_alphas)
        lower = mu[..., None] - half_width
        upper = mu[..., None] + half_width
        return torch.stack([lower, upper], dim=-2)

    def sample(self, y_pred: Tensor, n_members: int) -> TensorBTSCM:
        """Draw ``n_members`` per-site samples from the calibrated marginals."""
        mu, sigma = self._calibrated_moments(y_pred)
        eps = torch.randn(*mu.shape, n_members, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * eps
