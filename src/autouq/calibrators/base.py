import abc
from collections.abc import Callable, Sequence

import torch

from autouq.types import Tensor, TensorBTSCM, TensorBTSIA


# Example Algorithm
# 1. Fit Model to training data
# .   - No API needed
# 2. Fit calibration data x_c
#    - Get predictions (e.g. means): f(x_c). No API needed, entry point
#    - calculate absolute residuals: r_c = |y_c - f(x_c)|
#    - For a given alpha (target coverage) calculate
#           q = np.ceil((n+1)*(1-alpha))/n quantile of {r_c}'s
#      where n = len(r_c).
# 3. Get conformalized predictions
#     - calculate f(x_t), y_pred_t
#     - conformalize(y_pred_t) = y_pred_t ± q
class Calibrator(abc.ABC):
    """Calibrator base class."""

    def __init__(self, spatial_dims: Sequence[int]):
        # TODO: consider spatial handling
        self.spatial_dims = spatial_dims

    @abc.abstractmethod
    def calibrate(self, y_true: Tensor, y_pred: Tensor): ...

    # TODO: what should we have for predict here?
    def predict(self, y_pred: Tensor, alphas: float | Sequence[float]) -> Tensor:  # noqa: ARG002
        # TODO: replace with impl
        return torch.tensor(0)


class ConformalCalibrator(Calibrator, abc.ABC):
    """Conformal calibrator base class."""

    @abc.abstractmethod
    def _score_fn(self) -> Callable: ...

    def cache_scores(self, scores: Tensor):
        self.scores = scores

    def calibrate(self, y_true: Tensor, y_pred: Tensor):
        scores = self._score_fn()(y_true, y_pred)
        self.cache_scores(scores)

    def q_hat(self, alpha: float) -> Tensor:  # noqa: ARG002
        # TODO: add impl, placeholder
        return torch.tensor(0)

    @abc.abstractmethod
    def _predict(
        self, y_pred: Tensor, alpha: float | Sequence[float]
    ) -> TensorBTSIA:  # (B, T, *S, C, 2, n_alphas)?
        pass

    def predict(
        self, y_pred: Tensor, alphas: float | Sequence[float]
    ) -> TensorBTSIA:  # (B, T, *S, C, 2, n_alphas)?
        alphas = [alphas] if isinstance(alphas, float) else alphas
        # TODO: consider time dimension regarding the exchangeability assumption
        # should time be in batch dim ?
        # (see 2.4: https://arxiv.org/abs/2408.09881)
        # - e.g. could have helper functions for splitting into chunks
        # - also could have validation methods for checking the assumption
        return self._predict(y_pred, alphas)


# Target
class ConformalizedQuantileRegression(ConformalCalibrator):
    """Conformalized Quantile Regression base class."""


class AbsoluteErrorResidual(ConformalCalibrator):
    """Absolute Error Residual base class."""

    def _score_fn(self) -> Callable:
        # TODO: Calculate absolute residuals: r_c = |y_c - f(x_c)|
        return lambda _: _


class StandardDeviation(ConformalCalibrator):
    """Standard Deviation base class."""


class Ensemble(ConformalCalibrator):
    """Ensemble base class."""


class SamplableCalibrator(Calibrator, abc.ABC):
    """Marginal calibrator that also exposes a samplable predictive.

    Extends the :class:`Calibrator` interval contract with :meth:`sample`: a
    calibrator whose calibrated per-site predictive is a full distribution (not
    only an interval) can draw an ensemble from it. That ensemble is what a
    :class:`DependenceCalibrator` reorders. Interval-only calibrators (e.g. the
    conformal family) do not provide it.
    """

    @abc.abstractmethod
    def sample(
        self,
        y_pred: Tensor,
        n_members: int,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorBTSCM:
        """Draw ``n_members`` samples per site from the calibrated marginals.

        Parameters
        ----------
        y_pred
            Raw forecast to calibrate and sample from, shape ``(B, T, *S, C, M)``.
        n_members
            Number of ensemble members to draw.
        generator
            Optional :class:`torch.Generator` for reproducible draws; ``None``
            uses the global RNG.

        Returns
        -------
        TensorBTSCM
            Samples ``(B, T, *S, C, n_members)``, independent across sites.
        """


class DependenceCalibrator(abc.ABC):
    """Restore cross-site dependence onto calibrated marginal samples.

    Deliberately *not* a :class:`Calibrator`: it has no ``calibrate`` or
    ``predict`` and cannot produce a calibrated forecast by itself. It is a pure
    sample-to-sample transform that reorders an ensemble of independently drawn
    calibrated marginals to follow a dependence template's per-site rank order,
    so it only runs downstream of a marginal calibrator (see
    :class:`ComposedCalibrator`).
    """

    @abc.abstractmethod
    def apply(self, marginal_samples: Tensor, template: Tensor) -> Tensor:
        """Reorder ``marginal_samples`` to ``template``'s per-site rank order.

        Parameters
        ----------
        marginal_samples
            Independently drawn calibrated marginals, members on the last axis.
        template
            Dependence template of the same shape; only its per-site ranks are
            used.

        Returns
        -------
        Tensor
            The reordered ensemble: each site's marginal preserved exactly, the
            template's rank dependence imposed.
        """


class ComposedCalibrator(Calibrator):
    """A marginal calibrator followed by a dependence restorer.

    The two-phase post-processing pipeline: a :class:`SamplableCalibrator`
    (e.g. EMOS) calibrates each per-site marginal, then a
    :class:`DependenceCalibrator` (e.g. ECC) reorders sampled marginals to a
    template so the ensemble is jointly coherent again. Because the dependence
    step preserves each marginal exactly, :meth:`predict` is identical to the
    marginal's -- the joint structure the pipeline adds is visible only through
    :meth:`sample`.

    Parameters
    ----------
    marginal
        The per-site marginal calibrator (must be samplable).
    dependence
        The dependence restorer applied to the sampled marginals. It must
        treat the last axis as the ensemble members (matching
        :meth:`SamplableCalibrator.sample`'s output) -- e.g. the default
        ``ECC()`` with ``member_dim=-1``.
    """

    def __init__(self, marginal: SamplableCalibrator, dependence: DependenceCalibrator):
        super().__init__(marginal.spatial_dims)
        self.marginal = marginal
        self.dependence = dependence

    def calibrate(self, y_true: Tensor, y_pred: Tensor):
        """Fit the marginal calibrator; the dependence step is fit-free."""
        self.marginal.calibrate(y_true, y_pred)

    def predict(self, y_pred: Tensor, alphas: float | Sequence[float]) -> TensorBTSIA:
        """Per-site intervals, identical to the marginal's (reorder-invariant)."""
        return self.marginal.predict(y_pred, alphas)

    def sample(
        self,
        y_pred: Tensor,
        n_members: int | None = None,
        template: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorBTSCM:
        """Draw a jointly coherent ensemble: marginal draws reordered to a template.

        Parameters
        ----------
        y_pred
            Raw forecast, shape ``(B, T, *S, C, M)``.
        n_members
            Members to draw; defaults to the template's member count
            (size-preserving reorder).
        template
            Dependence template; defaults to ``y_pred`` itself (ensemble copula
            coupling onto the raw forecast's dependence).
        generator
            Optional :class:`torch.Generator` for reproducible marginal draws.

        Returns
        -------
        TensorBTSCM
            The coherent calibrated ensemble.
        """
        tmpl = y_pred if template is None else template
        n = tmpl.shape[-1] if n_members is None else n_members
        if n != tmpl.shape[-1]:
            msg = (
                f"n_members ({n}) must match the template's member count "
                f"({tmpl.shape[-1]}); the dependence reorder is size-preserving."
            )
            raise ValueError(msg)
        marginals = self.marginal.sample(y_pred, n, generator=generator)
        return self.dependence.apply(marginals, tmpl)
