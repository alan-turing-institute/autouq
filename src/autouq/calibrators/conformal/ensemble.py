from collections.abc import Sequence
from enum import StrEnum

import torch

from autouq.calibrators.conformal.conformal import ConformalCalibrator
from autouq.calibrators.grouping import validate_alpha, validate_alphas
from autouq.types import TensorBNC, TensorBNCM, TensorBNIA, TensorNC


class _EnsembleMode(StrEnum):
    QUANTILE = "quantile"
    STD = "std"


class Ensemble(ConformalCalibrator[TensorBNCM, TensorBNC, TensorNC]):
    """Conformal calibrator for ensemble forecasts.

    The final ``pred`` dimension indexes ensemble members. ``mode="quantile"``
    conformalizes an empirical ensemble quantile interval, while ``mode="std"``
    calibrates normalized residuals around the ensemble mean.

    In addition to the whole-tensor :meth:`calibrate`/:meth:`predict` API,
    this calibrator supports streaming one ensemble member at a time via
    :meth:`reset_stream`/:meth:`update_stream`/:meth:`finalize_stream_score` (for a
    calibration example) or :meth:`finalize_stream_predict` (for a
    prediction example) - useful when the full ``(*, channel, ensemble)``
    tensor for one example is too large to materialize at once. ``mode="std"``
    streams with no member retention at all (Welford's online algorithm);
    ``mode="quantile"`` still needs every member for ``torch.quantile``, but
    can tile that computation over a chunk of the leading axes via
    ``chunk_size``/``chunk_dim`` so the full stack is never combined at once.

    Args:
        temporal_dim: Optional index of the temporal dimension in tensors passed
            to the calibrator.
        spatial_dims: Optional indices of spatial dimensions in tensors passed
            to the calibrator.
        mode: Ensemble interval construction mode. Supported values are
            ``"quantile"`` and ``"std"``.
        ensemble_alpha: Miscoverage level used for the empirical ensemble
            quantile interval in ``"quantile"`` mode.
        min_scale: Minimum scale used to stabilize ``"std"`` mode.
    """

    def __init__(
        self,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
        mode: str = "quantile",
        ensemble_alpha: float = 0.1,
        min_scale: float = 1e-8,
    ):
        super().__init__(temporal_dim=temporal_dim, spatial_dims=spatial_dims)
        try:
            self.mode = _EnsembleMode(mode)
        except ValueError as exc:
            msg = f"mode must be 'quantile' or 'std', got {mode!r}."
            raise ValueError(msg) from exc
        validate_alpha(ensemble_alpha, name="ensemble_alpha")
        if min_scale <= 0:
            msg = f"min_scale must be positive, got {min_scale}."
            raise ValueError(msg)

        self.ensemble_alpha = ensemble_alpha
        self.min_scale = min_scale
        self._online_true: TensorBNC | None = None
        self._reset_member_accumulator()

    # ------------------------------------------------------------------
    # Whole-tensor API
    # ------------------------------------------------------------------

    def _score(self, true: TensorBNC, pred: TensorBNCM) -> TensorBNC:
        ensemble = self._validate_ensemble_predictions(pred)
        if true.shape != ensemble.shape[:-1]:
            msg = (
                "true must match the ensemble pred shape without the "
                f"ensemble dimension; got {tuple(true.shape)} and expected "
                f"{tuple(ensemble.shape[:-1])}."
            )
            raise ValueError(msg)
        return self._combine_score(true, self._summary(ensemble))

    def _predict(self, pred: TensorBNCM, alphas: Sequence[float]) -> TensorBNIA:
        scores = self._calibration_scores()
        ensemble = self._validate_ensemble_predictions(pred)
        if ensemble.shape[1:-1] != scores.shape[1:]:
            msg = (
                "pred must match the calibrated trailing shape; "
                f"got {tuple(ensemble.shape[1:-1])} and expected "
                f"{tuple(scores.shape[1:])}."
            )
            raise ValueError(msg)
        return self._combine_interval(self._summary(ensemble), alphas)

    def _validate_ensemble_predictions(self, pred: TensorBNCM) -> TensorBNCM:
        if pred.ndim < 2 or pred.shape[-1] < 2:
            msg = (
                "Ensemble predictions must include a final ensemble dimension "
                f"with at least 2 members; got shape {tuple(pred.shape)}."
            )
            raise ValueError(msg)
        return pred

    def _summary(self, ensemble: TensorBNCM) -> tuple[TensorBNC, TensorBNC]:
        """Return the mode-appropriate ensemble summary.

        ``(lower, upper)`` for ``"quantile"``, ``(center, scale)`` for
        ``"std"``.
        """
        if self.mode is _EnsembleMode.QUANTILE:
            return self._quantile_interval(ensemble)
        return self._mean_and_scale(ensemble)

    def _quantile_interval(self, ensemble: TensorBNCM) -> tuple[TensorBNC, TensorBNC]:
        lower_q = self.ensemble_alpha / 2
        upper_q = 1 - lower_q
        lower = torch.quantile(ensemble, lower_q, dim=-1)
        upper = torch.quantile(ensemble, upper_q, dim=-1)
        return lower, upper

    def _mean_and_scale(self, ensemble: TensorBNCM) -> tuple[TensorBNC, TensorBNC]:
        center = ensemble.mean(dim=-1)
        scale = ensemble.std(dim=-1, correction=0).clamp_min(self.min_scale)
        return center, scale

    def _combine_score(
        self, true: TensorBNC, summary: tuple[TensorBNC, TensorBNC]
    ) -> TensorBNC:
        if self.mode is _EnsembleMode.QUANTILE:
            lower, upper = summary
            return torch.maximum(lower - true, true - upper)
        center, scale = summary
        return torch.abs(true - center) / scale

    def _combine_interval(
        self, summary: tuple[TensorBNC, TensorBNC], alphas: Sequence[float]
    ) -> TensorBNIA:
        intervals = []
        if self.mode is _EnsembleMode.QUANTILE:
            lower, upper = summary
            for alpha in alphas:
                score_quantile = self.score_quantile(alpha).to(device=lower.device)
                intervals.append(
                    torch.stack(
                        (lower - score_quantile, upper + score_quantile), dim=-1
                    )
                )
        else:
            center, scale = summary
            for alpha in alphas:
                score_quantile = self.score_quantile(alpha).to(device=center.device)
                intervals.append(
                    torch.stack(
                        (
                            center - score_quantile * scale,
                            center + score_quantile * scale,
                        ),
                        dim=-1,
                    )
                )
        return torch.stack(intervals, dim=-1)

    # ------------------------------------------------------------------
    # Streaming interface — one ensemble member at a time
    # ------------------------------------------------------------------

    def _reset_member_accumulator(self) -> None:
        if self.mode is _EnsembleMode.QUANTILE:
            self._online_members: list[TensorBNC] = []
        else:
            self._online_count = 0
            self._online_mean: TensorBNC | None = None
            self._online_m2: TensorBNC | None = None

    def _accumulate_member(self, member: TensorBNC) -> None:
        if self.mode is _EnsembleMode.QUANTILE:
            self._online_members.append(member)
            return
        # Welford's online algorithm: running mean and sum-of-squared-deviations,
        # updated one member at a time with no retention.
        self._online_count += 1
        if self._online_mean is None:
            self._online_mean = member.clone()
            self._online_m2 = torch.zeros_like(member)
            return
        assert self._online_m2 is not None
        delta = member - self._online_mean
        self._online_mean = self._online_mean + delta / self._online_count
        self._online_m2 = self._online_m2 + delta * (member - self._online_mean)

    def _finalize_member_accumulator(
        self, chunk_size: int | None, chunk_dim: int
    ) -> tuple[TensorBNC, TensorBNC]:
        if self.mode is _EnsembleMode.QUANTILE:
            n_members = len(self._online_members)
            if n_members < 2:
                msg = (
                    "Ensemble predictions must include at least 2 members; "
                    f"got {n_members}."
                )
                raise ValueError(msg)
            if chunk_size is not None:
                if chunk_size <= 0:
                    msg = (
                        "chunk_size must be a positive integer or None, "
                        f"got {chunk_size}."
                    )
                    raise ValueError(msg)
                summary = self._quantile_interval_chunked(
                    self._online_members, chunk_size, chunk_dim
                )
            else:
                ensemble = torch.stack(self._online_members, dim=-1)
                summary = self._quantile_interval(ensemble)
            self._online_members = []
            return summary

        if self._online_count < 2:
            msg = (
                "Ensemble predictions must include at least 2 members; "
                f"got {self._online_count}."
            )
            raise ValueError(msg)
        # count >= 2 guarantees _accumulate_member has set both by now.
        assert self._online_mean is not None
        assert self._online_m2 is not None
        scale = torch.sqrt(self._online_m2 / self._online_count).clamp_min(
            self.min_scale
        )
        center = self._online_mean
        self._online_mean = None
        self._online_m2 = None
        self._online_count = 0
        return center, scale

    def _quantile_interval_chunked(
        self, members: list[TensorBNC], chunk_size: int, chunk_dim: int
    ) -> tuple[TensorBNC, TensorBNC]:
        """Tile the quantile-interval computation over ``chunk_dim``.

        Chunks of ``chunk_size`` at a time, so the full ``(*, channel,
        ensemble)`` stack over all retained members is never assembled at
        once - only one ``(*chunk, channel, ensemble)`` slice at a time.
        Mirrors fastnet's ``_finalize_over_gridpoint_chunks`` (chunking over
        the gridpoint axis), generalized to a caller-specified axis.
        """
        template = members[0]
        dim = chunk_dim if chunk_dim >= 0 else template.ndim + chunk_dim
        total = template.shape[dim]
        lower = torch.empty_like(template)
        upper = torch.empty_like(template)
        slicer: list[slice | int] = [slice(None)] * template.ndim
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            slicer[dim] = slice(start, end)
            idx = tuple(slicer)
            ensemble_chunk = torch.stack([member[idx] for member in members], dim=-1)
            lower_chunk, upper_chunk = self._quantile_interval(ensemble_chunk)
            lower[idx] = lower_chunk
            upper[idx] = upper_chunk
        return lower, upper

    def reset_stream(self) -> None:
        """Reset the online per-example ensemble accumulator.

        Call once before streaming members for a new calibration or
        prediction example via :meth:`update_stream`.
        """
        self._online_true = None
        self._reset_member_accumulator()

    def update_stream(self, member: TensorBNC, true: TensorBNC | None = None) -> None:
        """Accumulate one ensemble member's prediction for the current example.

        Args:
            member: One ensemble member's prediction, without an ensemble
                dimension.
            true: Target for this example, identical across members - only
                needs to be passed once (e.g. on the first call). Required
                before :meth:`finalize_stream_score` (a calibration example);
                omit it when the example is prediction-only
                (:meth:`finalize_stream_predict`).
        """
        if true is not None and self._online_true is None:
            self._online_true = true
        self._accumulate_member(member)

    def finalize_stream_score(
        self, chunk_size: int | None = None, chunk_dim: int = -2
    ) -> TensorNC:
        """Return this calibration example's conformity score.

        Combines the members streamed via :meth:`update_stream` (and the
        ``true`` passed alongside them) into the same score
        :meth:`calibrate` would compute from the whole tensor. Feed the
        result to :meth:`accumulate_score` to add it to the calibration
        bank::

            calibrator.reset_stream()
            for member in date_members:
                calibrator.update_stream(member, true=date_true)
            score = calibrator.finalize_stream_score(chunk_size=2048)
            calibrator.accumulate_score(score)

        Args:
            chunk_size: If set, tile the ensemble-summary computation over
                ``chunk_dim`` in chunks of this size (``mode="quantile"``
                only - ``mode="std"`` is already fully incremental and never
                needs this).
            chunk_dim: Axis of the per-member tensor to tile over when
                ``chunk_size`` is set. Defaults to the axis immediately
                before ``channel`` (typically a spatial axis).
        """
        summary = self._finalize_member_accumulator(chunk_size, chunk_dim)
        if self._online_true is None:
            msg = (
                "finalize_stream_score requires `true` to have been passed to "
                "update_stream for a calibration example."
            )
            raise ValueError(msg)
        score = self._combine_score(self._online_true, summary)
        self._online_true = None
        return score

    def finalize_stream_predict(
        self,
        alphas: float | Sequence[float],
        chunk_size: int | None = None,
        chunk_dim: int = -2,
    ) -> TensorBNIA:
        """Return calibrated prediction intervals for this streamed example.

        The prediction-time counterpart of :meth:`finalize_stream_score`: combines
        the members streamed via :meth:`update_stream` into the same
        intervals :meth:`predict` would compute from the whole tensor. No
        ``true`` is required (or used, even if passed to `update_stream`).

        Args:
            alphas: One or more miscoverage levels in ``(0, 1)``.
            chunk_size: See :meth:`finalize_stream_score`.
            chunk_dim: See :meth:`finalize_stream_score`.
        """
        alpha_values = validate_alphas(alphas)
        summary = self._finalize_member_accumulator(chunk_size, chunk_dim)
        return self._combine_interval(summary, alpha_values)
