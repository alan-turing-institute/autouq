"""Shared toy problems for the EMOS + ECC acceptance tests.

The three ``test_composed_*`` modules all probe one phenomenon from different
angles, so the machinery they share lives here.

The phenomenon
--------------
A forecast head can have *correct marginals* yet the *wrong dependence* across
cells (sites and/or lead times). Each per-cell interval is then well calibrated,
but any *aggregate* over a dependence axis -- a regional total summed over sites,
or an accumulated total summed over lead times -- is badly under-covered,
because the variance of a sum of correlated variables depends on their
covariance, which an independence-assuming ensemble does not encode. Ensemble
copula coupling (ECC) repairs this by reordering the calibrated marginals onto a
*template* ensemble that carries the true dependence -- so only a correctly
correlated template restores aggregate coverage; a white (independent) one does
not.

Each individual toy differs only in *where* the dependence lives (space, time,
or both). Everything shared -- covariance kernels, correlated Gaussian draws,
canonical-layout reshaping, and the aggregate-coverage diagnostic -- lives here.

Tensor layout
-------------
The canonical truth field is ``(B, T, S, C)`` and an ensemble is
``(B, T, S, C, M)``: batch, time/lead, a single spatial axis, channel, and
ensemble member (see ``autouq.types``). Every toy uses one spatial axis and one
channel (``C = 1``); batch ``B`` indexes independent calibration/test cases. The
dependence axis a toy sums over is therefore the time axis ``1`` or the spatial
axis ``2``.
"""

import torch

# ---------------------------------------------------------------------------
# Covariance kernels
# ---------------------------------------------------------------------------
# Each returns a unit-diagonal correlation matrix over the integer index
# 0..n-1. The Cholesky factor of one of these maps i.i.d. standard normals to
# draws carrying the intended cross-cell dependence (see ``correlated_normal``).


def ar_cov(n: int, rho: float) -> torch.Tensor:
    """AR(1) correlation over a 1-D index: ``corr(i, j) = rho ** |i - j|``.

    Models lead-to-lead persistence along a forecast rollout: the geometric
    decay of a stationary first-order autoregressive process. Positive-definite
    with unit diagonal for ``0 <= rho < 1``.
    """
    i = torch.arange(n)
    return rho ** (i[:, None] - i[None, :]).abs().float()


def ring_cov(n: int, length_scale: float) -> torch.Tensor:
    """Squared-exponential correlation on a ring of ``n`` equally spaced sites.

    Distance is the *circular* (wrap-around) separation
    ``min(|i - j|, n - |i - j|)``, so sites ``0`` and ``n - 1`` are neighbours.
    The diagonal is 1 by construction. On the ring metric this kernel is only
    positive-*semi*-definite, so callers make it strictly positive-definite --
    either a small diagonal jitter or :func:`psd_correlation` -- before taking a
    Cholesky factor.
    """
    i = torch.arange(n)
    d = (i[:, None] - i[None, :]).abs()
    d = torch.minimum(d, n - d).float()  # wrap-around distance on the ring
    return torch.exp(-0.5 * (d / length_scale) ** 2)


def psd_correlation(sigma: torch.Tensor) -> torch.Tensor:
    """Project a symmetric matrix onto the nearest unit-diagonal correlation.

    Clips the eigenvalues to a small positive floor (making the result strictly
    positive-definite, so it admits a Cholesky factor) and rescales rows and
    columns back to a unit diagonal. Needed because the ring kernel -- and any
    Kronecker product built from it -- is not guaranteed positive-definite.
    """
    w, v = torch.linalg.eigh(sigma)
    sigma = (v * w.clamp(min=1e-4)) @ v.T
    d = torch.sqrt(torch.diag(sigma))
    return sigma / torch.outer(d, d)


# ---------------------------------------------------------------------------
# Correlated Gaussian draws
# ---------------------------------------------------------------------------


def correlated_normal(chol: torch.Tensor, batch: int, members: int) -> torch.Tensor:
    """Draw ``(batch, dim, members)`` Gaussians with covariance ``chol @ chol.T``.

    ``chol`` is the lower-triangular Cholesky factor of a ``dim x dim``
    covariance over the (possibly flattened) cell index; draws are i.i.d. across
    ``batch`` cases and across ensemble ``members``. Reshape the ``dim`` axis
    into the desired ``(T, S)`` grid at the call site.
    """
    dim = chol.shape[0]
    z = torch.randn(batch, dim, members)
    return torch.einsum("ij,bjm->bim", chol, z)


# ---------------------------------------------------------------------------
# Canonical layout
# ---------------------------------------------------------------------------
# The toys build ``(B, T, S)`` fields (with a singleton on whichever of T/S they
# do not exercise); these append the trailing channel axis to reach the
# canonical layout the calibrators expect.


def as_truth(field: torch.Tensor) -> torch.Tensor:
    """Append the singleton channel axis: ``(B, T, S) -> (B, T, S, C=1)``."""
    return field[..., None]


def as_ensemble(field: torch.Tensor) -> torch.Tensor:
    """Insert the singleton channel axis: ``(B, T, S, M) -> (B, T, S, C=1, M)``."""
    return field[..., None, :]


# ---------------------------------------------------------------------------
# Coverage diagnostic
# ---------------------------------------------------------------------------


def collection_coverage(
    ensemble: torch.Tensor, y_true: torch.Tensor, axis: int, level: float = 0.90
) -> float:
    """Empirical coverage of the aggregate over ``axis`` at central ``level``.

    Sums the ensemble and the truth over a dependence axis (``axis = 1`` for the
    temporal total over leads, ``axis = 2`` for the spatial total over sites),
    forms the central ``level`` interval of the summed ensemble from its member
    quantiles, and returns the fraction of batch cases in which the summed truth
    lands inside. This is the discriminating quantity: it responds to the
    cross-cell correlation the ensemble encodes, to which per-cell marginals are
    blind.

    Args:
        ensemble: ``(B, T, S, C, M)`` calibrated ensemble.
        y_true: ``(B, T, S, C)`` ground truth.
        axis: dependence axis to sum over (``1`` = time/lead, ``2`` = space).
        level: central coverage level of the interval (default ``0.90``).

    Returns:
        The empirical coverage of the aggregate, in ``[0, 1]``.
    """
    agg_members = ensemble.sum(dim=axis)  # collapse dependence axis; members last
    agg_true = y_true.sum(dim=axis)
    lo = torch.quantile(agg_members, (1 - level) / 2, dim=-1)
    hi = torch.quantile(agg_members, (1 + level) / 2, dim=-1)
    inside = (agg_true >= lo) & (agg_true <= hi)
    return inside.float().mean().item()
