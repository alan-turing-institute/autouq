"""Calibrators: marginal, dependence, composed, and conformal post-hoc UQ methods."""

from autouq.calibrators.base import (
    Calibrator,
    ComposedCalibrator,
    DependenceCalibrator,
    SamplableCalibrator,
)
from autouq.calibrators.conformal import (
    AbsoluteErrorResidual,
    ConformalCalibrator,
    ConformalizedQuantileRegression,
    Ensemble,
    StandardDeviation,
)
from autouq.calibrators.ecc import ECC
from autouq.calibrators.emos import EMOS, EMOSECC
from autouq.calibrators.grouping import AxisRole
from autouq.calibrators.rcps import (
    EnsembleRCPS,
    HoeffdingBound,
    RCPSCalibrator,
    RiskBound,
    ScaledIntervalRCPS,
)

__all__ = [
    "ECC",
    "EMOS",
    "EMOSECC",
    "AbsoluteErrorResidual",
    "AxisRole",
    "Calibrator",
    "ComposedCalibrator",
    "ConformalCalibrator",
    "ConformalizedQuantileRegression",
    "DependenceCalibrator",
    "Ensemble",
    "EnsembleRCPS",
    "HoeffdingBound",
    "RCPSCalibrator",
    "RiskBound",
    "SamplableCalibrator",
    "ScaledIntervalRCPS",
    "StandardDeviation",
]
