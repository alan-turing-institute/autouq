import abc
from collections.abc import Sequence
from typing import Generic, TypeVar

import torch

from autouq.types import Tensor, TensorBNC, TensorBNIA, TensorBTSCM

PredT = TypeVar("PredT", bound=Tensor)


class Calibrator(Generic[PredT], abc.ABC):
    """Base class for prediction calibrators.

    Args:
        temporal_dim: Optional index of the temporal dimension in tensors passed
            to the calibrator.
        spatial_dims: Optional indices of spatial dimensions in tensors passed
            to the calibrator.
    """

    def __init__(
        self,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        self.temporal_dim = temporal_dim
        self.spatial_dims = spatial_dims

    @abc.abstractmethod
    def calibrate(self, true: TensorBNC, pred: PredT) -> None: ...


class SamplableCalibrator(Calibrator[Tensor], abc.ABC):
    """Marginal calibrator that also exposes a samplable predictive.

    Extends the :class:`Calibrator` contract with :meth:`predict` (central
    intervals) and :meth:`sample`: a calibrator whose calibrated per-site
    predictive is a full distribution (not only an interval) can draw an
    ensemble from it. That ensemble is what a :class:`DependenceCalibrator`
    reorders. Interval-only calibrators (e.g. the conformal family) do not
    provide it.
    """

    @abc.abstractmethod
    def predict(self, pred: TensorBTSCM, alphas: float | Sequence[float]) -> TensorBNIA:
        """Return central calibrated intervals at each miscoverage ``alpha``.

        Args:
            pred: Raw forecast to calibrate, shape ``(B, T, *S, C, M)``.
            alphas: One or more miscoverage levels in ``(0, 1)``.

        Returns:
            Intervals ``(B, T, *S, C, 2, n_alphas)`` (lower/upper on the
            penultimate axis).
        """

    @abc.abstractmethod
    def sample(
        self,
        pred: TensorBTSCM,
        n_members: int,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorBTSCM:
        """Draw ``n_members`` samples per site from the calibrated marginals.

        Args:
            pred: Raw forecast to calibrate and sample from, shape
                ``(B, T, *S, C, M)``.
            n_members: Number of ensemble members to draw.
            generator: Optional :class:`torch.Generator` for reproducible draws;
                ``None`` uses the global RNG.

        Returns:
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

        Args:
            marginal_samples: Independently drawn calibrated marginals, members
                on the last axis.
            template: Dependence template of the same shape; only its per-site
                ranks are used.

        Returns:
            The reordered ensemble: each site's marginal preserved exactly, the
            template's rank dependence imposed.
        """


class ComposedCalibrator(Calibrator[Tensor]):
    """A marginal calibrator followed by a dependence restorer.

    The two-phase post-processing pipeline: a :class:`SamplableCalibrator`
    (e.g. EMOS) calibrates each per-site marginal, then a
    :class:`DependenceCalibrator` (e.g. ECC) reorders sampled marginals to a
    template so the ensemble is jointly coherent again. Because the dependence
    step preserves each marginal exactly, :meth:`predict` is identical to the
    marginal's -- the joint structure the pipeline adds is visible only through
    :meth:`sample`.

    Args:
        marginal: The per-site marginal calibrator (must be samplable).
        dependence: The dependence restorer applied to the sampled marginals. It
            must treat the last axis as the ensemble members (matching
            :meth:`SamplableCalibrator.sample`'s output) -- e.g. the default
            ``ECC()`` with ``member_dim=-1``.
    """

    def __init__(self, marginal: SamplableCalibrator, dependence: DependenceCalibrator):
        super().__init__(spatial_dims=marginal.spatial_dims)
        self.marginal = marginal
        self.dependence = dependence

    def calibrate(self, true: TensorBNC, pred: TensorBTSCM) -> None:
        """Fit the marginal calibrator; the dependence step is fit-free."""
        self.marginal.calibrate(true, pred)

    def predict(self, pred: TensorBTSCM, alphas: float | Sequence[float]) -> TensorBNIA:
        """Per-site intervals, identical to the marginal's (reorder-invariant)."""
        return self.marginal.predict(pred, alphas)

    def sample(
        self,
        pred: TensorBTSCM,
        n_members: int | None = None,
        template: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorBTSCM:
        """Draw a jointly coherent ensemble: marginal draws reordered to a template.

        Args:
            pred: Raw forecast, shape ``(B, T, *S, C, M)``.
            n_members: Members to draw; defaults to the template's member count
                (size-preserving reorder).
            template: Dependence template; defaults to ``pred`` itself (ensemble
                copula coupling onto the raw forecast's dependence).
            generator: Optional :class:`torch.Generator` for reproducible
                marginal draws.

        Returns:
            The coherent calibrated ensemble ``(B, T, *S, C, n_members)``.
        """
        tmpl = pred if template is None else template
        n = tmpl.shape[-1] if n_members is None else n_members
        if n != tmpl.shape[-1]:
            msg = (
                f"n_members ({n}) must match the template's member count "
                f"({tmpl.shape[-1]}); the dependence reorder is size-preserving."
            )
            raise ValueError(msg)
        marginals = self.marginal.sample(pred, n, generator=generator)
        return self.dependence.apply(marginals, tmpl)
