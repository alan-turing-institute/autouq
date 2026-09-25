import math

import torch

from autouq.calibrators.optimize import minimize_per_group


def _autograd_derivatives(group_loss):
    """``group_derivatives`` for ``group_loss``, from autograd on the group sum."""

    def derivatives(theta):
        theta = theta.detach().requires_grad_(True)
        losses = group_loss(theta)
        (grad,) = torch.autograd.grad(losses.sum(), theta, create_graph=True)
        hess = torch.stack(
            [
                torch.autograd.grad(grad[k].sum(), theta, retain_graph=True)[0]
                for k in range(theta.shape[0])
            ]
        )
        return losses.detach(), grad.detach(), hess.detach()

    return derivatives


def test_minimize_per_group_solves_bounded_quadratics_independently():
    # separable quadratics whose curvatures span six orders of magnitude across
    # groups; the last coordinate is bounded below by 0 and its unconstrained
    # minimum is negative in half the groups, so those must stop on the bound
    curvature = torch.tensor(
        [[1e-3, 1.0, 1e3], [1e3, 1e-3, 1.0], [1.0, 1e3, 1e-3], [10.0, 10.0, 10.0]],
        dtype=torch.float64,
    ).T  # (P, G)
    centre = torch.tensor(
        [[0.5, -2.0, 3.0, 1.0], [-1.0, 4.0, 0.0, 2.0], [0.7, -0.3, 2.0, -5.0]],
        dtype=torch.float64,
    )  # (P, G)
    lower = torch.tensor([-math.inf, -math.inf, 0.0])

    def group_loss(theta):
        return 1.0 + (curvature * (theta - centre) ** 2).sum(dim=0)

    theta, converged = minimize_per_group(
        group_loss,
        _autograd_derivatives(group_loss),
        torch.ones_like(centre),
        lower,
        max_iter=50,
        tol=1e-12,
    )

    assert converged.all()
    expected = torch.maximum(centre, lower.to(centre)[:, None])
    torch.testing.assert_close(theta, expected, rtol=0, atol=1e-8)


def test_minimize_per_group_handles_nonconvex_groups():
    # shifted Rosenbrock valleys with different minima (a, a^2) and stiffness b;
    # the Hessian is indefinite along the way, so plain Newton steps would ascend
    a = torch.tensor([-1.0, 0.5, 2.0, 1.0], dtype=torch.float64)
    b = torch.tensor([1.0, 100.0, 10.0, 100.0], dtype=torch.float64)

    def group_loss(theta):
        x, y = theta
        return 1.0 + (a - x) ** 2 + b * (y - x**2) ** 2

    theta0 = torch.tensor([[-1.2] * 4, [1.0] * 4], dtype=torch.float64)
    theta, converged = minimize_per_group(
        group_loss,
        _autograd_derivatives(group_loss),
        theta0,
        torch.full((2,), -math.inf),
        max_iter=200,
        tol=1e-14,
    )

    assert converged.all()
    torch.testing.assert_close(theta, torch.stack([a, a**2]), rtol=0, atol=1e-6)


def test_minimize_per_group_reports_groups_short_of_tol():
    # one Newton iteration solves the quadratic group, not the Rosenbrock one
    def group_loss(theta):
        x, y = theta
        rosenbrock = 1.0 + (1.0 - x[0]) ** 2 + 100.0 * (y[0] - x[0] ** 2) ** 2
        quadratic = 1.0 + (x[1] - 3.0) ** 2 + (y[1] + 1.0) ** 2
        return torch.stack([rosenbrock, quadratic])

    theta0 = torch.tensor([[-1.2, 0.0], [1.0, 0.0]], dtype=torch.float64)
    _, converged = minimize_per_group(
        group_loss,
        _autograd_derivatives(group_loss),
        theta0,
        torch.full((2,), -math.inf),
        max_iter=2,
        tol=1e-12,
    )

    assert converged.tolist() == [False, True]
