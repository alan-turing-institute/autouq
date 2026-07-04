"""EMOS (nonhomogeneous Gaussian regression) and the composed EMOS+ECC pipeline."""

from collections.abc import Sequence

import torch
from torch.nn.functional import softplus

from autouq.calibrators.base import ComposedCalibrator, SamplableCalibrator
from autouq.calibrators.ecc import ECC
from autouq.calibrators.grouping import (
    AxisRole,
    group_layout,
    group_location_scale,
    normalize_roles,
    validate_alphas,
)
from autouq.calibrators.mathutils import STD_NORMAL, gaussian_crps, inv_softplus
from autouq.types import Tensor, TensorBNIA, TensorBTSC, TensorBTSCM

_VAR_FLOOR = 1e-12


class EMOS(SamplableCalibrator):
    r"""Ensemble Model Output Statistics (nonhomogeneous Gaussian regression).

    A parametric marginal calibrator. For each group (see ``per``) it fits an
    affine map from the raw ensemble's mean :math:`\bar{x}` and variance
    :math:`s^2` to a calibrated Gaussian predictive,

    .. math::
        \mu_{\mathrm{cal}} = \beta_0 + \beta_1 \bar{x}, \qquad
        \sigma^2_{\mathrm{cal}} = \gamma_0 + \gamma_1 s^2
        \quad (\gamma_0, \gamma_1 \ge 0),

    by minimising the mean closed-form Gaussian CRPS over the calibration set.
    The non-negativity of :math:`\gamma_0, \gamma_1` is enforced with a softplus
    link. It subsumes pure variance inflation as the constrained case
    :math:`\beta_0 = 0, \beta_1 = 1, \gamma_0 = 0`.

    Independent coefficients are fit along the axes named in ``per`` (default the
    time/lead axis) and pooled over the rest, so the same model calibrates per
    lead time, per spatial site, or per channel -- or any combination. Inputs are
    standardised per group before fitting and de-standardised on output, so the
    optimisation is well conditioned regardless of the data's scale. As a
    :class:`SamplableCalibrator` it can also draw an ensemble from the calibrated
    marginals (:meth:`sample`), for composition with a dependence calibrator.

    Which axes are calibrated is controlled entirely by ``per``; the spatial and
    channel axis positions are inferred from the forecast's rank at
    :meth:`calibrate` time under the canonical ``(B, T, *S, C)`` layout, so no
    axis sizes or indices need to be supplied up front.

    Args:
        per: Axis roles that get their own coefficients, as :class:`AxisRole`
            members (bare strings are also accepted and normalised). Default
            ``(AxisRole.TIME,)`` -- one fit per lead. Use e.g.
            ``per=(AxisRole.SPACE,)`` to calibrate per site or
            ``per=(AxisRole.TIME, AxisRole.SPACE)`` for both. Batch is always
            pooled.
        max_iter: L-BFGS iterations for the per-group CRPS fit.
        lr: L-BFGS learning rate.
    """

    def __init__(
        self,
        *,
        per: Sequence[AxisRole | str] = (AxisRole.TIME,),
        max_iter: int = 100,
        lr: float = 1.0,
    ):
        super().__init__()
        self.per = normalize_roles(per)
        self.max_iter = max_iter
        self.lr = lr
        self._beta0: Tensor | None = None
        self._beta1: Tensor | None = None
        self._raw_gamma0: Tensor | None = None
        self._raw_gamma1: Tensor | None = None
        self._loc: Tensor | None = None
        self._scale: Tensor | None = None

    @staticmethod
    def _ensemble_mean_var(pred: TensorBTSCM) -> tuple[Tensor, Tensor]:
        """Raw ensemble mean and sample variance over the member axis."""
        n_members = pred.shape[-1]
        if n_members < 2:
            msg = (
                "EMOS needs an ensemble of at least 2 members to estimate "
                f"spread; got {n_members} on the last axis."
            )
            raise ValueError(msg)
        return pred.mean(dim=-1), pred.var(dim=-1, unbiased=True)

    def calibrate(self, true: TensorBTSC, pred: TensorBTSCM) -> None:
        """Fit per-group affine coefficients by minimising mean Gaussian CRPS."""
        xbar, s2 = self._ensemble_mean_var(pred)
        group_dims, pshape, view = group_layout(xbar.shape, self.per)

        # standardise per group so the fit runs on O(1) quantities at any scale
        loc, scale = group_location_scale(true, group_dims)
        loc_v, scale_v = loc.view(view), scale.view(view)
        xbar_s = (xbar - loc_v) / scale_v
        s2_s = s2 / scale_v**2
        y_s = (true - loc_v) / scale_v

        def _leaf(t: Tensor) -> Tensor:
            """Return a grad-tracking leaf parameter on the working device/dtype."""
            return t.to(device=xbar.device, dtype=xbar.dtype).requires_grad_(True)

        beta0 = _leaf(torch.zeros(pshape))
        beta1 = _leaf(torch.ones(pshape))
        raw_g0 = _leaf(torch.full(pshape, inv_softplus(1e-3)))
        raw_g1 = _leaf(torch.full(pshape, inv_softplus(1.0)))
        params = [beta0, beta1, raw_g0, raw_g1]
        optimizer = torch.optim.LBFGS(
            params, lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            mu = beta0.view(view) + beta1.view(view) * xbar_s
            sigma2 = softplus(raw_g0).view(view) + softplus(raw_g1).view(view) * s2_s
            sigma = torch.sqrt(sigma2 + _VAR_FLOOR)
            loss = gaussian_crps(mu, sigma, y_s).mean()
            loss.backward()
            return loss

        optimizer.step(closure)

        if not all(torch.isfinite(p).all() for p in params):
            msg = (
                "EMOS fit produced non-finite coefficients; L-BFGS did not "
                "converge. Check the calibration data and `lr`/`max_iter`."
            )
            raise RuntimeError(msg)

        self._beta0 = beta0.detach()
        self._beta1 = beta1.detach()
        self._raw_gamma0 = raw_g0.detach()
        self._raw_gamma1 = raw_g1.detach()
        self._loc = loc.detach()
        self._scale = scale.detach()

    def _calibrated_mean_std(self, pred: TensorBTSCM) -> tuple[Tensor, Tensor]:
        """Calibrated per-site Gaussian mean and standard deviation."""
        if (
            self._beta0 is None
            or self._beta1 is None
            or self._raw_gamma0 is None
            or self._raw_gamma1 is None
            or self._loc is None
            or self._scale is None
        ):
            msg = "EMOS.calibrate must be called before predict/sample."
            raise RuntimeError(msg)
        xbar, s2 = self._ensemble_mean_var(pred)
        _, pshape, view = group_layout(xbar.shape, self.per)
        fitted_shape = tuple(self._beta0.shape)
        if pshape != fitted_shape:
            msg = (
                f"EMOS was fit with calibration-axis sizes {fitted_shape} but got "
                f"{pshape} at predict/sample time; the calibrated axes "
                f"(per={self.per}) must match the fit."
            )
            raise ValueError(msg)
        dev = xbar.device
        beta0 = self._beta0.to(dev).view(view)
        beta1 = self._beta1.to(dev).view(view)
        raw_g0 = self._raw_gamma0.to(dev).view(view)
        raw_g1 = self._raw_gamma1.to(dev).view(view)
        loc = self._loc.to(dev).view(view)
        scale = self._scale.to(dev).view(view)
        xbar_s = (xbar - loc) / scale
        s2_s = s2 / scale**2
        mu_s = beta0 + beta1 * xbar_s
        sigma2_s = softplus(raw_g0) + softplus(raw_g1) * s2_s
        sigma_s = torch.sqrt(sigma2_s + _VAR_FLOOR)
        return scale * mu_s + loc, scale * sigma_s

    def predict(self, pred: TensorBTSCM, alphas: float | Sequence[float]) -> TensorBNIA:
        """Central calibrated Gaussian intervals at each miscoverage ``alpha``."""
        alphas = validate_alphas(alphas)
        mu, sigma = self._calibrated_mean_std(pred)
        quantiles = torch.tensor(
            [1.0 - a / 2.0 for a in alphas], device=mu.device, dtype=mu.dtype
        )
        zs = STD_NORMAL.icdf(quantiles)  # (n_alphas,)
        half_width = sigma[..., None] * zs  # (..., n_alphas)
        lower = mu[..., None] - half_width
        upper = mu[..., None] + half_width
        return torch.stack([lower, upper], dim=-2)

    def sample(
        self,
        pred: TensorBTSCM,
        n_members: int,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorBTSCM:
        """Draw ``n_members`` per-site samples from the calibrated marginals."""
        mu, sigma = self._calibrated_mean_std(pred)
        eps = torch.randn(
            *mu.shape,
            n_members,
            device=mu.device,
            dtype=mu.dtype,
            generator=generator,
        )
        return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * eps


class EMOSECC(ComposedCalibrator):
    r"""EMOS marginals coupled with ECC dependence, as a single calibrator.

    A ready-made instance of the marginal-plus-dependence pipeline (see
    :class:`~autouq.calibrators.base.ComposedCalibrator`): it calibrates each
    per-group Gaussian marginal with :class:`EMOS` (nonhomogeneous Gaussian
    regression), then restores joint dependence across sites and lead times --
    any non-member axes the template carries -- with
    :class:`~autouq.calibrators.ecc.ECC` (ensemble copula coupling), which has no
    axis-specific logic. Use it like any other
    :class:`~autouq.calibrators.base.Calibrator` -- ``calibrate`` to fit, then
    ``predict`` for marginal intervals or ``sample`` for a jointly coherent
    ensemble; the two-stage composition is handled internally.

    Args:
        **emos_kwargs: Keyword arguments for :class:`EMOS` (e.g. ``per``,
            ``max_iter``, ``lr``).

    Examples:
        Construct and use it like any other calibrator -- ``calibrate`` to fit,
        then ``predict`` or ``sample``::

            model = EMOSECC()  # or EMOSECC(per=(AxisRole.SPACE,)) for per-site
            # true: (B, T, *S, C); pred: (B, T, *S, C, M)
            model.calibrate(true, pred)
            intervals = model.predict(pred, alphas=0.1)  # 90% marginal intervals
            ensemble = model.sample(pred, n_members=32)  # coherent calibrated ensemble
    """

    def __init__(self, **emos_kwargs):
        super().__init__(EMOS(**emos_kwargs), ECC())
