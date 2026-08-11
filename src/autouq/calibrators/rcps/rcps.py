import abc
import math
from collections.abc import Sequence
from typing import Generic, cast

import torch

from autouq.calibrators.base import Calibrator, PredT
from autouq.calibrators.grouping import validate_open_unit_interval
from autouq.calibrators.rcps.bounds import HoeffdingBound, RiskBound
from autouq.metrics import CoverageLoss, RiskLoss
from autouq.types import Tensor, TensorB, TensorBNC, TensorBNI, TensorBNIAD


def _validate_risk_levels(alphas: float | Sequence[float]) -> list[float]:
    if isinstance(alphas, float):
        values = [alphas]
    elif isinstance(alphas, int):
        msg = "risk level alpha must be a float or a sequence of floats."
        raise TypeError(msg)
    else:
        values = [float(alpha) for alpha in alphas]
    if not values:
        msg = "at least one risk level alpha is required."
        raise ValueError(msg)
    if any(not math.isfinite(alpha) or alpha < 0 for alpha in values):
        msg = "risk level alpha must be finite and non-negative."
        raise ValueError(msg)
    return values


def _validate_deltas(deltas: float | Sequence[float]) -> list[float]:
    if isinstance(deltas, float):
        values = [deltas]
    elif isinstance(deltas, int):
        msg = "delta must be a float or a sequence of floats."
        raise TypeError(msg)
    else:
        values = [float(delta) for delta in deltas]
    if not values:
        msg = "at least one delta is required."
        raise ValueError(msg)
    for delta in values:
        validate_open_unit_interval(delta, name="delta")
    return values


class RCPSCalibrator(Calibrator[PredT], Generic[PredT], abc.ABC):
    """Base class for risk-controlling prediction-set calibrators.

    Subclasses define how their prediction type produces a nested interval
    family. This base class evaluates the configured loss, applies the risk
    bound, and searches for the smallest feasible interval scale.

    Args:
        alphas: Target risk levels. A scale is fitted for each alpha/delta pair.
        deltas: Probabilities with which risk control may fail over the random
            calibration-set draw.
        loss: Per-example, non-increasing risk loss. Defaults to
            :class:`CoverageLoss`.
        bound: Risk upper confidence bound. It must preserve the loss ordering
            used by the binary search. Defaults to :class:`HoeffdingBound`.
        search_tolerance: Absolute tolerance for binary search over lambda.
        max_search_steps: Maximum upper expansion and binary-search iterations.
        temporal_dim: Optional temporal dimension index.
        spatial_dims: Optional spatial dimension indices.
    """

    def __init__(
        self,
        alphas: float | Sequence[float],
        deltas: float | Sequence[float],
        loss: RiskLoss | None = None,
        bound: RiskBound | None = None,
        *,
        search_tolerance: float = 1e-4,
        max_search_steps: int = 64,
        temporal_dim: int | None = None,
        spatial_dims: Sequence[int] | None = None,
    ):
        super().__init__(temporal_dim=temporal_dim, spatial_dims=spatial_dims)
        if not math.isfinite(search_tolerance) or search_tolerance <= 0:
            msg = "search_tolerance must be finite and greater than zero."
            raise ValueError(msg)
        if max_search_steps < 1:
            msg = "max_search_steps must be at least 1."
            raise ValueError(msg)

        self.alphas = tuple(_validate_risk_levels(alphas))
        self.deltas = tuple(_validate_deltas(deltas))
        self.loss = CoverageLoss() if loss is None else loss
        self.bound = HoeffdingBound() if bound is None else bound
        self.search_tolerance = search_tolerance
        self.max_search_steps = max_search_steps
        self.lambda_hats: dict[tuple[float, float], float] = {}
        self._calibration_true: TensorBNC | None = None
        self._calibration_pred: PredT | None = None

    @abc.abstractmethod
    def _validate_calibration_prediction(self, true: TensorBNC, pred: PredT) -> None:
        """Validate prediction structure against calibration targets."""

    @abc.abstractmethod
    def _validate_test_prediction(self, pred: PredT, calibration_pred: PredT) -> None:
        """Validate prediction structure against cached calibration data."""

    @abc.abstractmethod
    def _prediction_set(self, pred: PredT, lambda_: float, alpha: float) -> TensorBNI:
        """Construct the nested prediction interval at ``lambda_``."""

    def calibrate(self, true: TensorBNC, pred: PredT) -> None:
        """Fit scales for the configured Cartesian product of alpha and delta."""
        self._validate_calibration_prediction(true, pred)
        if true.shape[0] == 0:
            msg = "calibration data must contain at least one batch item."
            raise ValueError(msg)
        if true.device != pred.device:
            msg = "true and pred must be on the same device."
            raise ValueError(msg)
        if not torch.isfinite(true).all() or not torch.isfinite(pred).all():
            msg = "true and pred must contain only finite values."
            raise ValueError(msg)

        self._calibration_true = true.detach()
        self._calibration_pred = cast("PredT", pred.detach())
        fitted_scales = {}
        try:
            for alpha in self.alphas:
                for delta in self.deltas:
                    fitted_scales[(alpha, delta)] = self._find_lambda_hat(alpha, delta)
        except Exception:
            self._calibration_true = None
            self._calibration_pred = None
            self.lambda_hats.clear()
            raise
        self.lambda_hats = fitted_scales

    def _calibration_data(self) -> tuple[TensorBNC, PredT]:
        if self._calibration_true is None or self._calibration_pred is None:
            msg = "Calibrator must be calibrated before computing an RCPS scale."
            raise RuntimeError(msg)
        return self._calibration_true, self._calibration_pred

    def _losses(self, lambda_: float, alpha: float) -> TensorB:
        true, pred = self._calibration_data()
        losses = self.loss(self._prediction_set(pred, lambda_, alpha), true)
        if not isinstance(losses, Tensor):
            msg = "loss must return a torch.Tensor."
            raise TypeError(msg)
        if losses.shape != (true.shape[0],):
            msg = (
                "loss must return one value per calibration example with shape "
                f"({true.shape[0]},); got {tuple(losses.shape)}."
            )
            raise ValueError(msg)
        if not torch.isfinite(losses).all() or (losses < 0).any():
            msg = "loss must return finite, non-negative values."
            raise ValueError(msg)
        return losses

    def risk_upper_bound(self, lambda_: float, alpha: float, delta: float) -> float:
        """Return the configured upper confidence bound on risk."""
        if not math.isfinite(lambda_) or lambda_ < 0:
            msg = "lambda_ must be finite and non-negative."
            raise ValueError(msg)
        alpha_value = _validate_risk_levels(alpha)[0]
        validate_open_unit_interval(delta, name="delta")
        value = self.bound(self._losses(lambda_, alpha_value), delta)
        if not math.isfinite(value):
            msg = "bound must return a finite float."
            raise ValueError(msg)
        return value

    def lambda_hat(self, alpha: float, delta: float) -> float:
        """Return the fitted scale for one configured risk/error-level pair."""
        alpha_value = _validate_risk_levels(alpha)[0]
        validate_open_unit_interval(delta, name="delta")
        key = (alpha_value, delta)
        if alpha_value not in self.alphas or delta not in self.deltas:
            msg = "alpha and delta must be configured when constructing the calibrator."
            raise ValueError(msg)
        self._calibration_data()
        return self.lambda_hats[key]

    def _find_lambda_hat(self, alpha: float, delta: float) -> float:
        if self.risk_upper_bound(lambda_=0.0, alpha=alpha, delta=delta) <= alpha:
            return 0.0

        lower = 0.0
        upper = 1.0
        for _ in range(self.max_search_steps):
            if self.risk_upper_bound(lambda_=upper, alpha=alpha, delta=delta) <= alpha:
                break
            lower = upper
            upper *= 2
        else:
            msg = (
                "Could not find a risk-controlling interval scale; check the "
                "loss/bound assumptions and prediction-set widths."
            )
            raise RuntimeError(msg)

        for _ in range(self.max_search_steps):
            if upper - lower <= self.search_tolerance:
                break
            midpoint = (lower + upper) / 2
            if (
                self.risk_upper_bound(lambda_=midpoint, alpha=alpha, delta=delta)
                <= alpha
            ):
                upper = midpoint
            else:
                lower = midpoint

        return upper

    def predict(self, pred: PredT) -> TensorBNIAD:
        """Return intervals for the configured alpha/delta Cartesian product."""
        _, calibration_pred = self._calibration_data()
        self._validate_test_prediction(pred, calibration_pred)
        if not torch.isfinite(pred).all():
            msg = "pred must contain only finite values."
            raise ValueError(msg)

        alpha_intervals = []
        for alpha in self.alphas:
            delta_intervals = [
                self._prediction_set(pred, self.lambda_hat(alpha, delta), alpha)
                for delta in self.deltas
            ]
            alpha_intervals.append(torch.stack(delta_intervals, dim=-1))
        return torch.stack(alpha_intervals, dim=-2)
