"""Elementary math helpers shared by the calibrators.

Small, self-contained numerical primitives (softplus inverse, standard-normal
density, closed-form Gaussian CRPS) factored out of the individual calibrators
so they can be reused and tested in isolation.
"""

import math

import torch
from torch.distributions import Normal

from autouq.types import Tensor

STD_NORMAL = Normal(0.0, 1.0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def inv_softplus(x: float) -> float:
    """Inverse softplus, for initialising a softplus-constrained parameter."""
    return math.log(math.expm1(x))


def std_normal_pdf(z: Tensor) -> Tensor:
    """Standard-normal density evaluated elementwise."""
    return torch.exp(-0.5 * z * z) / _SQRT_2PI


def gaussian_crps(mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
    """Closed-form CRPS of a Gaussian predictive (Gneiting et al., 2005)."""
    z = (y - mu) / sigma
    cdf = STD_NORMAL.cdf(z)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * std_normal_pdf(z) - _INV_SQRT_PI)
