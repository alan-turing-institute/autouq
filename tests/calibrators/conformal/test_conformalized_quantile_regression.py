import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from autouq.calibrators import ConformalizedQuantileRegression


@pytest.fixture
def calibrator():
    # Tests use channels-last tensors with one singleton spatial axis.
    return ConformalizedQuantileRegression(
        alphas=0.4,
        temporal_dim=1,
        spatial_dims=(2,),
    )


def _interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    score_quantile: torch.Tensor,
) -> torch.Tensor:
    return torch.stack((lower - score_quantile, upper + score_quantile), dim=-1)


def test_calibrate_caches_cqr_scores(calibrator):
    # Shape: (calibration_examples=2, time=2, spatial=1, channels=1).
    true = torch.tensor(
        [
            [[[2.0]], [[8.0]]],
            [[[5.0]], [[3.0]]],
        ]
    )
    lower = torch.tensor(
        [
            [[[1.0]], [[4.0]]],
            [[[6.0]], [[1.0]]],
        ]
    )
    upper = torch.tensor(
        [
            [[[3.0]], [[6.0]]],
            [[[8.0]], [[4.0]]],
        ]
    )
    pred = torch.stack((lower, upper), dim=-1)

    calibrator.calibrate(true, pred)

    expected_scores = torch.tensor(
        [
            [[[-1.0]], [[2.0]]],
            [[[1.0]], [[-1.0]]],
        ]
    )
    torch.testing.assert_close(calibrator.scores, expected_scores)


def test_predict_returns_conformalized_quantile_intervals(
    calibrator,
    finite_sample_scores,
    finite_sample_score_quantiles,
):
    # For n=4, alpha=0.4 selects the 3rd smallest score. This correction is
    # applied to the lower/upper quantile pair for alpha=0.4.
    calibrator.cache_scores(finite_sample_scores)

    lower = torch.tensor(
        [
            [[[100.0]], [[200.0]]],
            [[[300.0]], [[400.0]]],
        ]
    )
    upper = lower + 10.0
    pred = torch.stack((lower, upper), dim=-1)

    intervals = calibrator.predict(pred, alphas=0.4)

    expected = _interval(
        lower,
        upper,
        finite_sample_score_quantiles[0.4],
    ).unsqueeze(-1)

    assert intervals.shape == (2, 2, 1, 1, 2, 1)
    torch.testing.assert_close(intervals, expected)


def test_multi_alpha_cqr_selects_matching_quantile_pairs_and_scores(
    finite_sample_scores,
    finite_sample_score_quantiles,
):
    calibrator = ConformalizedQuantileRegression(
        alphas=[0.4, 0.2],
        temporal_dim=1,
        spatial_dims=(2,),
    )
    scores = torch.stack((finite_sample_scores, finite_sample_scores + 100.0), dim=-1)
    calibrator.cache_scores(scores)

    lower_alpha_04 = torch.tensor(
        [
            [[[100.0]], [[200.0]]],
            [[[300.0]], [[400.0]]],
        ]
    )
    lower_alpha_02 = lower_alpha_04 - 10.0
    upper_alpha_04 = lower_alpha_04 + 20.0
    upper_alpha_02 = lower_alpha_02 + 40.0
    lower = torch.stack((lower_alpha_04, lower_alpha_02), dim=-1)
    upper = torch.stack((upper_alpha_04, upper_alpha_02), dim=-1)
    pred = torch.stack((lower, upper), dim=-2)

    intervals = calibrator.predict(pred, alphas=[0.4, 0.2])

    expected = torch.stack(
        (
            _interval(
                lower_alpha_04,
                upper_alpha_04,
                finite_sample_score_quantiles[0.4],
            ),
            _interval(
                lower_alpha_02,
                upper_alpha_02,
                finite_sample_score_quantiles[0.2] + 100.0,
            ),
        ),
        dim=-1,
    )

    assert intervals.shape == (2, 2, 1, 1, 2, 2)
    torch.testing.assert_close(intervals, expected)


def test_multi_alpha_calibrate_caches_alpha_indexed_scores():
    calibrator = ConformalizedQuantileRegression(
        alphas=[0.4, 0.2],
        temporal_dim=1,
        spatial_dims=(2,),
    )
    true = torch.tensor([[[[10.0]]], [[[20.0]]]])
    lower = torch.tensor(
        [
            [[[[8.0, 7.0]]]],
            [[[[21.0, 19.0]]]],
        ]
    )
    upper = torch.tensor(
        [
            [[[[12.0, 13.0]]]],
            [[[[23.0, 24.0]]]],
        ]
    )
    pred = torch.stack((lower, upper), dim=-2)

    calibrator.calibrate(true, pred)

    expected_scores = torch.tensor(
        [
            [[[[-2.0, -3.0]]]],
            [[[[1.0, -1.0]]]],
        ]
    )
    torch.testing.assert_close(calibrator.scores, expected_scores)


def test_calibrate_rejects_invalid_quantile_prediction_shape(calibrator):
    with pytest.raises(BeartypeCallHintParamViolation):
        calibrator.calibrate(torch.ones(2, 2, 1, 1), torch.ones(2, 2, 1, 1))

    with pytest.raises(ValueError, match="quantile pred shape"):
        calibrator.calibrate(torch.ones(2, 3, 1, 1), torch.ones(2, 2, 1, 1, 2))


def test_calibrate_rejects_crossed_quantiles(calibrator):
    lower = torch.ones(2, 2, 1, 1) * 2.0
    upper = torch.ones(2, 2, 1, 1)
    pred = torch.stack((lower, upper), dim=-1)

    with pytest.raises(ValueError, match="Lower quantile"):
        calibrator.calibrate(torch.ones(2, 2, 1, 1), pred)


def test_predict_rejects_uncalibrated_or_mismatched_inputs(calibrator):
    pred = torch.ones(1, 2, 1, 1, 2)

    with pytest.raises(RuntimeError, match="must be calibrated"):
        calibrator.predict(pred, alphas=0.2)

    calibrator.cache_scores(torch.ones(4, 2, 1, 1))

    with pytest.raises(ValueError, match="trailing shape"):
        calibrator.predict(torch.ones(1, 3, 1, 1, 2), alphas=0.2)


def test_predict_rejects_unconfigured_alpha(calibrator):
    calibrator.cache_scores(torch.ones(4, 2, 1, 1))

    with pytest.raises(ValueError, match="not configured"):
        calibrator.predict(torch.ones(1, 2, 1, 1, 2), alphas=0.2)


def test_multi_alpha_cqr_rejects_missing_alpha_dimension():
    calibrator = ConformalizedQuantileRegression(
        alphas=[0.4, 0.2],
        temporal_dim=1,
        spatial_dims=(2,),
    )

    with pytest.raises(ValueError, match="multiple alphas"):
        calibrator.calibrate(torch.ones(2, 2, 1, 1), torch.ones(2, 2, 1, 1, 2))

    with pytest.raises(ValueError, match="one quantile pair"):
        calibrator.calibrate(torch.ones(2, 2, 1, 1), torch.ones(2, 2, 1, 1, 2, 1))
