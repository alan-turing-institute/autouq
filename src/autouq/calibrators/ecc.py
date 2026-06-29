"""Ensemble copula coupling / Schaake shuffle dependence calibrator."""

import torch

from autouq.calibrators.base import DependenceCalibrator
from autouq.types import Tensor


class ECC(DependenceCalibrator):
    """Ensemble copula coupling (and the Schaake shuffle).

    Reorders independently drawn calibrated marginal samples to follow a
    dependence template's per-site rank order. At every site the output members
    are a permutation of the input members -- so each marginal is preserved
    exactly -- and that permutation matches the template's per-site ranks, so the
    output inherits the template's cross-site / temporal dependence (its copula).
    The template is the raw forecast ensemble (ensemble copula coupling) or
    historical trajectories (the Schaake shuffle); only its ranks are used. The
    reorder is size-preserving: samples and template must share a member count.

    Parameters
    ----------
    member_dim
        Axis indexing the ensemble members (default last, the ``TensorBTSCM``
        convention).
    """

    def __init__(self, member_dim: int = -1):
        self.member_dim = member_dim

    def apply(self, marginal_samples: Tensor, template: Tensor) -> Tensor:
        """Reorder ``marginal_samples`` to ``template``'s per-site rank order."""
        if marginal_samples.shape != template.shape:
            msg = (
                "marginal_samples and template must have the same shape; got "
                f"{tuple(marginal_samples.shape)} vs {tuple(template.shape)}."
            )
            raise ValueError(msg)
        dim = self.member_dim
        n_members = marginal_samples.shape[dim]
        if n_members < 2:
            msg = f"ECC needs at least 2 members on dim {dim}; got {n_members}."
            raise ValueError(msg)
        sorted_marginals = torch.sort(marginal_samples, dim=dim).values
        template_ranks = template.argsort(dim=dim).argsort(dim=dim)
        return torch.gather(sorted_marginals, dim, template_ranks)
