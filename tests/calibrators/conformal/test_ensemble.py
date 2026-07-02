import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from autouq.calibrators import Ensemble


def _interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    score_quantile: torch.Tensor,
) -> torch.Tensor:
    return torch.stack((lower - score_quantile, upper + score_quantile), dim=-1)


@pytest.fixture
def calibrator():
    # Tests use channels-last tensors with one singleton spatial axis.
    return Ensemble(
        temporal_dim=1,
        spatial_dims=(2,),
        mode="quantile",
        ensemble_alpha=0.5,
    )


@pytest.fixture
def std_calibrator():
    # Tests use channels-last tensors with one singleton spatial axis.
    return Ensemble(temporal_dim=1, spatial_dims=(2,), mode="std")


def test_quantile_mode_calibrate_caches_interval_scores(calibrator):
    # Shape: (calibration_examples=2, time=1, spatial=1, channels=1, members=5).
    pred = torch.tensor(
        [
            [[[[0.0, 10.0, 20.0, 30.0, 40.0]]]],
            [[[[100.0, 110.0, 120.0, 130.0, 140.0]]]],
        ]
    )
    true = torch.tensor([[[[35.0]]], [[[105.0]]]])

    calibrator.calibrate(true, pred)

    torch.testing.assert_close(calibrator.scores, torch.ones(2, 1, 1, 1) * 5.0)


def test_quantile_mode_predicts_conformalized_ensemble_intervals(
    calibrator,
    finite_sample_scores,
    finite_sample_score_quantiles,
):
    calibrator.cache_scores(finite_sample_scores)
    base = torch.tensor(
        [
            [[[100.0]], [[200.0]]],
            [[[300.0]], [[400.0]]],
        ]
    )
    pred = torch.stack(
        [base, base + 10.0, base + 20.0, base + 30.0, base + 40.0],
        dim=-1,
    )

    intervals = calibrator.predict(pred, alphas=[0.4, 0.2])

    lower = base + 10.0
    upper = base + 30.0
    expected = torch.stack(
        (
            _interval(lower, upper, finite_sample_score_quantiles[0.4]),
            _interval(lower, upper, finite_sample_score_quantiles[0.2]),
        ),
        dim=-1,
    )

    assert intervals.shape == (2, 2, 1, 1, 2, 2)
    torch.testing.assert_close(intervals, expected)


def test_std_mode_calibrate_caches_normalized_scores(std_calibrator):
    pred = torch.tensor(
        [
            [[[[8.0, 12.0]]]],
            [[[[18.0, 22.0]]]],
        ]
    )
    true = torch.tensor([[[[13.0]]], [[[20.0]]]])

    std_calibrator.calibrate(true, pred)

    expected_scores = torch.tensor([[[[1.5]]], [[[0.0]]]])
    torch.testing.assert_close(std_calibrator.scores, expected_scores)


def test_std_mode_predicts_scaled_conformal_intervals(std_calibrator):
    scores = torch.tensor(
        [
            [[[1.0]]],
            [[[4.0]]],
            [[[2.0]]],
            [[[3.0]]],
        ]
    )
    std_calibrator.cache_scores(scores)
    pred = torch.tensor([[[[[8.0, 12.0]]]], [[[[18.0, 22.0]]]]])

    intervals = std_calibrator.predict(pred, alphas=0.4)

    expected = torch.tensor(
        [
            [[[[[4.0], [16.0]]]]],
            [[[[[14.0], [26.0]]]]],
        ]
    )
    assert intervals.shape == (2, 1, 1, 1, 2, 1)
    torch.testing.assert_close(intervals, expected)


def test_ensemble_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="mode"):
        Ensemble(temporal_dim=1, spatial_dims=(2,), mode="median")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ensemble_alpha"):
        Ensemble(temporal_dim=1, spatial_dims=(2,), ensemble_alpha=0.0)

    with pytest.raises(ValueError, match="min_scale"):
        Ensemble(temporal_dim=1, spatial_dims=(2,), min_scale=0.0)


def test_ensemble_rejects_invalid_prediction_shapes(calibrator):
    with pytest.raises(BeartypeCallHintParamViolation):
        calibrator.calibrate(torch.ones(2, 1, 1, 1), torch.ones(2))

    with pytest.raises(ValueError, match="at least 2 members"):
        calibrator.calibrate(torch.ones(2, 1, 1), torch.ones(2, 1, 1, 1))

    with pytest.raises(ValueError, match="without the ensemble dimension"):
        calibrator.calibrate(torch.ones(2, 2, 1, 1), torch.ones(2, 1, 1, 1, 2))

    calibrator.cache_scores(torch.ones(4, 2, 1, 1))
    with pytest.raises(ValueError, match="trailing shape"):
        calibrator.predict(torch.ones(1, 3, 1, 1, 2), alphas=0.2)
