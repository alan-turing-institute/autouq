"""Elementary math helpers shared by the calibrators.

Small, self-contained numerical primitives (standard-normal density,
closed-form Gaussian CRPS and its derivatives) factored out of the individual
calibrators so they can be reused and tested in isolation.
"""

import math
from typing import NamedTuple

import torch
from torch.distributions import Normal

from autouq.types import Tensor

STD_NORMAL = Normal(0.0, 1.0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def std_normal_pdf(z: Tensor) -> Tensor:
    """Standard-normal density evaluated elementwise."""
    return torch.exp(-0.5 * z * z) / _SQRT_2PI


def _crps_from_z(sigma: Tensor, z: Tensor, cdf: Tensor, pdf: Tensor) -> Tensor:
    """Gaussian CRPS from the standardised error and its normal cdf and pdf."""
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI)


def gaussian_crps(mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
    """Closed-form CRPS of a Gaussian predictive (Gneiting et al., 2005)."""
    z = (y - mu) / sigma
    return _crps_from_z(sigma, z, STD_NORMAL.cdf(z), std_normal_pdf(z))


class GaussianCRPSDerivatives(NamedTuple):
    """Gaussian CRPS with its partial derivatives in the mean and the std."""

    value: Tensor
    d_mu: Tensor
    d_sigma: Tensor
    d_mu_mu: Tensor
    d_mu_sigma: Tensor
    d_sigma_sigma: Tensor


def gaussian_crps_derivatives(
    mu: Tensor, sigma: Tensor, y: Tensor
) -> GaussianCRPSDerivatives:
    r"""Closed-form Gaussian CRPS and its first and second derivatives.

    With :math:`z = (y - \mu) / \sigma`, standard-normal cdf :math:`\Phi` and
    density :math:`\phi`:

    .. math::
        \partial_\mu = 1 - 2\Phi(z), \quad
        \partial_\sigma = 2\phi(z) - 1/\sqrt{\pi}, \quad
        \partial_{\mu\mu} = 2\phi(z)/\sigma, \quad
        \partial_{\mu\sigma} = 2 z \phi(z)/\sigma, \quad
        \partial_{\sigma\sigma} = 2 z^2 \phi(z)/\sigma.
    """
    z = (y - mu) / sigma
    cdf = STD_NORMAL.cdf(z)
    pdf = std_normal_pdf(z)
    curvature = 2.0 * pdf / sigma
    return GaussianCRPSDerivatives(
        value=_crps_from_z(sigma, z, cdf, pdf),
        d_mu=1.0 - 2.0 * cdf,
        d_sigma=2.0 * pdf - _INV_SQRT_PI,
        d_mu_mu=curvature,
        d_mu_sigma=z * curvature,
        d_sigma_sigma=z * z * curvature,
    )
