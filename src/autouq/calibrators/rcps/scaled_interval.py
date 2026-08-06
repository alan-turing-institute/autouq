import torch

from autouq.calibrators.rcps.rcps import RCPSCalibrator
from autouq.types import TensorBNC, TensorBNCU, TensorBNI


class ScaledIntervalRCPS(RCPSCalibrator[TensorBNCU]):
    r"""RCPS for predicted centres and asymmetric uncertainty widths.

    The final prediction axis stores centre, lower width, and upper width. The
    nested interval is

    .. math::

        [f(X) - \lambda l(X), f(X) + \lambda u(X)].
    """

    @staticmethod
    def _validate_prediction(pred: TensorBNCU) -> None:
        if pred.ndim < 3 or pred.shape[-1] != 3:
            msg = (
                "pred must end in centre, lower-width, and upper-width "
                f"components; got shape {tuple(pred.shape)}."
            )
            raise ValueError(msg)
        if pred.shape[0] == 0:
            msg = "pred must contain at least one batch item."
            raise ValueError(msg)
        if (pred[..., 1:] < 0).any():
            msg = "lower and upper uncertainty widths must be non-negative."
            raise ValueError(msg)

    def _validate_calibration_prediction(
        self, true: TensorBNC, pred: TensorBNCU
    ) -> None:
        self._validate_prediction(pred)
        if pred.shape[:-1] != true.shape:
            msg = (
                "pred without its component axis must match true; got "
                f"{tuple(pred.shape[:-1])} and {tuple(true.shape)}."
            )
            raise ValueError(msg)

    def _validate_test_prediction(
        self, pred: TensorBNCU, calibration_pred: TensorBNCU
    ) -> None:
        self._validate_prediction(pred)
        if pred.shape[1:-1] != calibration_pred.shape[1:-1]:
            msg = (
                "pred must match the calibrated trailing shape before its "
                f"component axis; got {tuple(pred.shape[1:-1])} and expected "
                f"{tuple(calibration_pred.shape[1:-1])}."
            )
            raise ValueError(msg)

    def _prediction_set(
        self, pred: TensorBNCU, lambda_: float, alpha: float
    ) -> TensorBNI:
        del alpha
        centre, lower_width, upper_width = pred.unbind(dim=-1)
        return torch.stack(
            (centre - lambda_ * lower_width, centre + lambda_ * upper_width),
            dim=-1,
        )
