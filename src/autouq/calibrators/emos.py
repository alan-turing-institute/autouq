"""EMOS (nonhomogeneous Gaussian regression) and the composed EMOS+ECC pipeline."""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from autouq.calibrators.base import ComposedCalibrator, SamplableCalibrator
from autouq.calibrators.ecc import ECC
from autouq.calibrators.grouping import (
    AxisRole,
    group_layout,
    group_location_scale,
    normalize_roles,
    pooled_dims,
    validate_alphas,
)
from autouq.calibrators.mathutils import (
    STD_NORMAL,
    gaussian_crps,
    gaussian_crps_derivatives,
)
from autouq.calibrators.optimize import minimize_per_group
from autouq.types import Tensor, TensorBNIA, TensorBTSC, TensorBTSCM

_VAR_FLOOR = 1e-12
_FIT_DTYPE = torch.float64


def _emos_mean_std(
    beta0: Tensor,
    beta1: Tensor,
    gamma0: Tensor,
    gamma1: Tensor,
    xbar: Tensor,
    s2: Tensor,
) -> tuple[Tensor, Tensor]:
    """EMOS predictive mean and standard deviation from ensemble mean and variance."""
    return beta0 + beta1 * xbar, torch.sqrt(gamma0 + gamma1 * s2 + _VAR_FLOOR)


@dataclass(frozen=True)
class _GroupCRPS:
    """Each group's mean Gaussian CRPS as a function of its EMOS coefficients.

    Holds standardised data: the truth ``y``, the ensemble mean ``xbar`` and the
    ensemble variance ``s2`` scaled to mean one per group. ``theta`` stacks
    ``(beta0, beta1, gamma0, gamma1)``, one entry per group, along a leading axis
    of size 4; ``view`` lays a group's coefficient over the data, and the mean
    runs over the ``pooled`` axes.
    """

    y: Tensor
    xbar: Tensor
    s2: Tensor
    view: tuple[int, ...]
    pooled: tuple[int, ...]

    def _predictive(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and standard deviation at every data point."""
        beta0, beta1, gamma0, gamma1 = (p.view(self.view) for p in theta)
        return _emos_mean_std(beta0, beta1, gamma0, gamma1, self.xbar, self.s2)

    def __call__(self, theta: Tensor) -> Tensor:
        """Each group's mean CRPS."""
        return gaussian_crps(*self._predictive(theta), self.y).mean(dim=self.pooled)

    def derivatives(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Each group's mean CRPS with its gradient and Hessian in ``theta``.

        By the chain rule through ``mu = beta0 + beta1 * xbar`` and
        ``sigma = sqrt(gamma0 + gamma1 * s2 + floor)``: the mean coefficients
        enter through ``(1, xbar)``, the variance coefficients through
        ``(1, s2) / (2 sigma)``, whose own derivative contributes
        ``-(1, s2)(1, s2)^T / (4 sigma^3)`` to the variance block.
        """
        mu, sigma = self._predictive(theta)
        crps = gaussian_crps_derivatives(mu, sigma, self.y)
        half_inv_sigma = 0.5 / sigma
        d_var = crps.d_sigma * half_inv_sigma
        cross = crps.d_mu_sigma * half_inv_sigma
        var_curv = half_inv_sigma**2 * (crps.d_sigma_sigma - crps.d_sigma / sigma)
        x, s2 = self.xbar, self.s2

        def avg(t: Tensor) -> Tensor:
            return t.mean(dim=self.pooled)

        def moments(w: Tensor, a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            """Means of ``w``, ``w * a`` and ``w * a**2``."""
            wa = w * a
            return avg(w), avg(wa), avg(wa * a)

        grad = torch.stack(
            [avg(crps.d_mu), avg(crps.d_mu * x), avg(d_var), avg(d_var * s2)]
        )
        h00, h01, h11 = moments(crps.d_mu_mu, x)
        h22, h23, h33 = moments(var_curv, s2)
        h02, h03 = avg(cross), avg(cross * s2)
        cross_x = cross * x
        h12, h13 = avg(cross_x), avg(cross_x * s2)
        hess = torch.stack(
            [
                torch.stack([h00, h01, h02, h03]),
                torch.stack([h01, h11, h12, h13]),
                torch.stack([h02, h12, h22, h23]),
                torch.stack([h03, h13, h23, h33]),
            ]
        )
        return avg(crps.value), grad, hess


class EMOS(SamplableCalibrator):
    r"""Ensemble Model Output Statistics (nonhomogeneous Gaussian regression).

    A parametric marginal calibrator. For each group (see ``per``) it fits an
    affine map from the raw ensemble's mean :math:`\bar{x}` and variance
    :math:`s^2` to a calibrated Gaussian predictive,

    .. math::
        \mu_{\mathrm{cal}} = \beta_0 + \beta_1 \bar{x}, \qquad
        \sigma^2_{\mathrm{cal}} = \gamma_0 + \gamma_1 s^2
        \quad (\gamma_0, \gamma_1 \ge 0),

    by minimising the mean closed-form Gaussian CRPS over the calibration set,
    with :math:`\gamma_0, \gamma_1 \ge 0` imposed as bounds. It subsumes pure
    variance inflation as the constrained case
    :math:`\beta_0 = 0, \beta_1 = 1, \gamma_0 = 0`.

    Independent coefficients are fit along the axes named in ``per`` (default the
    time/lead axis) and pooled over the rest, so the same model calibrates per
    lead time, per spatial site, or per channel -- or any combination. Each group
    is its own four-parameter problem, and every group is solved to its own
    convergence (see :func:`~autouq.calibrators.optimize.minimize_per_group`), in
    float64. Inputs are standardised per group before fitting and de-standardised
    on output, and the ensemble variance is scaled to mean one per group, so the
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
        max_iter: Maximum Newton iterations; all groups iterate together.
        tol: Relative convergence tolerance: a group stops once the predicted
            decrease of its Newton step is below ``tol`` times its mean CRPS. A
            warning reports any group still short of it after ``max_iter``
            iterations.
    """

    def __init__(
        self,
        *,
        per: Sequence[AxisRole | str] = (AxisRole.TIME,),
        max_iter: int = 100,
        tol: float = 1e-10,
    ):
        super().__init__()
        self.per = normalize_roles(per)
        self.max_iter = max_iter
        self.tol = tol
        self._beta0: Tensor | None = None
        self._beta1: Tensor | None = None
        self._gamma0: Tensor | None = None
        self._gamma1: Tensor | None = None
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
        if not (torch.isfinite(true).all() and torch.isfinite(pred).all()):
            msg = "EMOS.calibrate needs finite `true` and `pred`; got NaN or inf."
            raise ValueError(msg)
        xbar, s2 = (m.to(_FIT_DTYPE) for m in self._ensemble_mean_var(pred))
        true = true.to(_FIT_DTYPE)
        group_dims, _, view = group_layout(xbar.shape, self.per)
        pooled = pooled_dims(xbar.ndim, group_dims)

        # standardise per group so the fit runs on O(1) quantities at any scale,
        # and scale the ensemble variance to mean one so gamma1 is O(1) as well;
        # rebinding the names frees the unstandardised copies before the fit
        loc, scale = group_location_scale(true, group_dims)
        loc_v, scale_v = loc.view(view), scale.view(view)
        xbar = (xbar - loc_v) / scale_v
        true = (true - loc_v) / scale_v
        s2 = s2 / scale_v**2
        # a group whose ensemble has collapsed has no spread to scale by; leaving
        # it at one keeps gamma1 from being divided back up by a floor at predict
        mean_s2 = s2.mean(dim=pooled)
        spread = torch.where(mean_s2 > 0, mean_s2, torch.ones_like(mean_s2))
        s2 = s2 / spread.view(view)
        objective = _GroupCRPS(true, xbar, s2, view, pooled)

        # start from the raw ensemble mean, with the predictive variance matched to
        # its mean squared error and split evenly between floor and spread terms
        half_mse = ((true - xbar) ** 2).mean(dim=pooled) / 2
        theta0 = torch.stack(
            [torch.zeros_like(half_mse), torch.ones_like(half_mse), half_mse, half_mse]
        )
        lower = torch.tensor([-math.inf, -math.inf, 0.0, 0.0])
        theta, converged = minimize_per_group(
            objective,
            objective.derivatives,
            theta0,
            lower,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        if not torch.isfinite(theta).all():
            msg = (
                "EMOS fit produced non-finite coefficients. Check the calibration "
                "data for NaN/inf values."
            )
            raise RuntimeError(msg)
        if not converged.all():
            warnings.warn(
                f"EMOS: {int((~converged).sum())} of {converged.numel()} groups did "
                f"not converge within max_iter={self.max_iter}; their coefficients "
                "may be sub-optimal. Raise `max_iter` or check the calibration data.",
                RuntimeWarning,
                stacklevel=2,
            )

        beta0, beta1, gamma0, gamma1 = theta
        self._beta0 = beta0
        self._beta1 = beta1
        self._gamma0 = gamma0
        self._gamma1 = gamma1 / spread  # per unit of s2 / scale**2, as predict uses
        self._loc = loc
        self._scale = scale

    def _calibrated_mean_std(self, pred: TensorBTSCM) -> tuple[Tensor, Tensor]:
        """Calibrated per-site Gaussian mean and standard deviation."""
        if (
            self._beta0 is None
            or self._beta1 is None
            or self._gamma0 is None
            or self._gamma1 is None
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
        beta0, beta1, gamma0, gamma1, loc, scale = (
            t.to(device=xbar.device, dtype=xbar.dtype).view(view)
            for t in (
                self._beta0,
                self._beta1,
                self._gamma0,
                self._gamma1,
                self._loc,
                self._scale,
            )
        )
        mu_s, sigma_s = _emos_mean_std(
            beta0, beta1, gamma0, gamma1, (xbar - loc) / scale, s2 / scale**2
        )
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
            ``max_iter``, ``tol``).

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
