"""Batched minimisation of many small, independent per-group problems.

A calibrator that fits its own coefficients for each group (lead time, site,
channel, ...) solves one small problem per group. Pooling them into a single
optimiser couples their stopping rules, iteration budget and curvature model, so
groups that differ from each other stall short of their own optimum. This module
solves every group at once, but each to its own convergence.
"""

from collections.abc import Callable

import torch

from autouq.types import Tensor

_ARMIJO_C = 1e-4
_MAX_HALVINGS = 40
_REL_FLOOR = 1e-12
_FRACTION_TO_BOUND = 0.9


def minimize_per_group(
    group_loss: Callable[[Tensor], Tensor],
    group_derivatives: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]],
    theta0: Tensor,
    lower: Tensor,
    *,
    max_iter: int,
    tol: float,
) -> tuple[Tensor, Tensor]:
    """Minimise independent per-group objectives with safeguarded Newton steps.

    ``group_loss`` maps parameters of shape ``(P, *G)`` -- ``P`` parameters for
    each of the ``*G`` groups -- to the per-group objectives, shape ``G``, where
    group ``g``'s objective depends only on ``theta[:, g]``. Each iteration takes,
    for every group, a Newton step on a positive-definite model of its Hessian,
    in which no bounded parameter covers more than 90% of its distance to its
    bound, then backtracks until the objective decreases enough. Iterates stay
    strictly inside the bounds and approach a bound geometrically when the
    optimum lies on it. A group stops once the predicted decrease of its step is
    below ``tol`` times its objective, so the objectives should be bounded away
    from zero (a CRPS is).

    Args:
        group_loss: Per-group objectives, shape ``G``.
        group_derivatives: Per-group objectives with their gradients, shape
            ``(P, *G)``, and Hessians, shape ``(P, P, *G)``.
        theta0: Starting point, shape ``(P, *G)``, strictly inside the bounds.
        lower: Lower bound for each parameter, shape ``(P,)``; ``-inf`` for none.
        max_iter: Maximum number of Newton iterations.
        tol: Relative tolerance on each group's predicted decrease.

    Returns:
        ``(theta, converged)``: the minimiser, shape ``(P, *G)``, and a boolean
        mask of shape ``G`` marking the groups that met ``tol`` at the last check.
    """
    n_params, group_shape = theta0.shape[0], theta0.shape[1:]
    lower = lower.to(theta0).view(1, n_params)

    def unflatten(x: Tensor) -> Tensor:
        """``(P, *G)`` parameters from flat ``(n_groups, P)`` ones."""
        return x.T.reshape(n_params, *group_shape)

    def losses_of(x: Tensor) -> Tensor:
        """Per-group objectives for flat ``(n_groups, P)`` parameters."""
        return group_loss(unflatten(x)).reshape(-1)

    x = theta0.reshape(n_params, -1).T.clone()
    converged = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for _ in range(max_iter):
        losses, grad, hess = group_derivatives(unflatten(x))
        losses = losses.reshape(-1)
        grad = grad.reshape(n_params, -1).T
        hess = hess.reshape(n_params, n_params, -1).permute(2, 0, 1)
        room = torch.where(torch.isfinite(lower), x - lower, torch.inf)
        step = _descent_step(grad, hess, room)
        converged = -(grad * step).sum(dim=-1) <= tol * losses.abs()
        if converged.all():
            break
        x = _backtrack(losses_of, x, losses, grad, step, todo=~converged)
    return unflatten(x), converged.reshape(group_shape)


def _descent_step(grad: Tensor, hess: Tensor, room: Tensor) -> Tensor:
    """Newton step on a positive-definite Hessian model, kept inside the bounds.

    No coordinate may move more than ``_FRACTION_TO_BOUND`` of its ``room``
    towards its bound. A coordinate that the Newton step would push past its
    bound while its own gradient also points there moves that fraction of the
    way, and the Newton step for the others is re-solved without it, so they
    are not computed as if it had moved further. If clipping still leaves a
    group with no descent (possible only through coupling between parameters),
    the group takes a diagonally scaled gradient step instead.
    """
    floor = -_FRACTION_TO_BOUND * room
    newton = _newton_step(grad, hess, torch.ones_like(grad, dtype=torch.bool))
    to_bound = (newton < floor) & (grad > 0)
    rows = to_bound.any(dim=-1)
    newton[rows] = _newton_step(grad[rows], hess[rows], ~to_bound[rows])
    step = torch.where(to_bound, floor, torch.maximum(newton, floor))

    diag = hess.diagonal(dim1=-2, dim2=-1).abs().clamp_min(torch.finfo(grad.dtype).tiny)
    fallback = torch.maximum(-grad / diag, floor)
    descends = (grad * step).sum(dim=-1) < 0
    return torch.where(descends[:, None], step, fallback)


def _newton_step(grad: Tensor, hess: Tensor, free: Tensor) -> Tensor:
    """Newton step over the ``free`` coordinates; the others stay put.

    The Hessian is scaled by its diagonal and its eigenvalues replaced by their
    absolute values, floored, so the model is positive definite however badly
    the parameters are scaled.
    """
    eye = torch.eye(grad.shape[-1], dtype=grad.dtype, device=grad.device)
    hess = torch.where(free[:, :, None] & free[:, None, :], hess, eye)
    grad = torch.where(free, grad, 0.0)
    scale = _floor_magnitude(hess.diagonal(dim1=-2, dim2=-1).abs()).rsqrt()
    evals, evecs = torch.linalg.eigh(hess * scale[:, :, None] * scale[:, None, :])
    mags = _floor_magnitude(evals.abs())
    scaled_grad = (scale * grad)[..., None]
    return -scale * (evecs @ ((evecs.mT @ scaled_grad) / mags[..., None]))[..., 0]


def _floor_magnitude(t: Tensor) -> Tensor:
    """Floor non-negative values at a fraction of their row's largest, and above 0."""
    t = t.clamp_min(_REL_FLOOR * t.amax(dim=-1, keepdim=True))
    return t.clamp_min(torch.finfo(t.dtype).tiny)


def _backtrack(
    losses_of: Callable[[Tensor], Tensor],
    x: Tensor,
    losses: Tensor,
    grad: Tensor,
    step: Tensor,
    todo: Tensor,
) -> Tensor:
    """Per-group Armijo backtracking along ``step``, halving the step size.

    Every trial is measured from the starting ``x``, so a group that has already
    accepted its step keeps evaluating that same (feasible) point.
    """
    size = torch.ones_like(losses)
    accepted = x
    for _ in range(_MAX_HALVINGS):
        trial = x + size[:, None] * step
        trial_losses = losses_of(trial)
        decrease = size * (grad * step).sum(dim=-1)
        sufficient = trial_losses <= losses + _ARMIJO_C * decrease
        accepted = torch.where((todo & sufficient)[:, None], trial, accepted)
        todo = todo & ~sufficient
        if not todo.any():
            break
        size = torch.where(todo, size / 2, size)
    return accepted
