import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from autouq.calibrators import AbsoluteErrorResidual


@pytest.fixture
def calibrator():
    # Tests use channels-last tensors with one singleton spatial axis.
    return AbsoluteErrorResidual(temporal_dim=1, spatial_dims=(2,))


@pytest.fixture
def pred():
    # Shape: (batch=2, time=2, spatial=1, channels=1). The large values make
    # interval offsets easy to inspect in conformal interval tests.
    return torch.tensor(
        [
            [[[100.0]], [[200.0]]],
            [[[300.0]], [[400.0]]],
        ]
    )


def _interval(center: torch.Tensor, score_quantile: torch.Tensor) -> torch.Tensor:
    return torch.stack((center - score_quantile, center + score_quantile), dim=-1)


def test_calibrate_caches_absolute_error_scores(calibrator):
    # Shape: (calibration_examples=2, time=2, spatial=1, channels=1). The
    # singleton spatial axis keeps the test aligned with the configured dims.
    true = torch.tensor(
        [
            [[[1.0]], [[4.0]]],
            [[[-2.0]], [[8.0]]],
        ]
    )
    pred = torch.tensor(
        [
            [[[0.5]], [[5.0]]],
            [[[1.0]], [[6.0]]],
        ]
    )

    calibrator.calibrate(true, pred)

    expected_scores = torch.tensor(
        [
            [[[0.5]], [[1.0]]],
            [[[3.0]], [[2.0]]],
        ]
    )
    torch.testing.assert_close(calibrator.scores, expected_scores)


def test_score_quantile_uses_finite_sample_order_statistic_per_cell(calibrator):
    # Shape: (calibration_examples=4, time=2, spatial=1, channels=1). The score
    # quantile is checked independently for each time/spatial/channel cell. For n=4,
    # alpha=0.4 selects the 3rd smallest score and alpha=0.2 selects the 4th.
    conformal_scores = torch.tensor(
        [
            [[[1.0]], [[10.0]]],
            [[[4.0]], [[40.0]]],
            [[[2.0]], [[20.0]]],
            [[[3.0]], [[30.0]]],
        ]
    )
    calibrator.cache_scores(conformal_scores)

    torch.testing.assert_close(
        calibrator.score_quantile(alpha=0.4),
        torch.tensor([[[3.0]], [[30.0]]]),
    )
    torch.testing.assert_close(
        calibrator.score_quantile(alpha=0.2),
        torch.tensor([[[4.0]], [[40.0]]]),
    )


def test_predict_returns_symmetric_intervals_for_each_alpha(
    calibrator,
    pred,
):
    # Shape: (calibration_examples=4, time=2, spatial=1, channels=1). This
    # mirrors the score quantile pattern, but includes the singleton spatial axis
    # implied by the configured dims. The output interval shape is
    # (batch, optional_dims..., channel, lower_upper, alphas).
    spatiotemporal_conformal_scores = torch.tensor(
        [
            [[[1.0]], [[10.0]]],
            [[[4.0]], [[40.0]]],
            [[[2.0]], [[20.0]]],
            [[[3.0]], [[30.0]]],
        ]
    )
    calibrator.cache_scores(spatiotemporal_conformal_scores)

    intervals = calibrator.predict(pred, alphas=[0.4, 0.2])

    score_quantile_alpha_04 = torch.tensor([[[3.0]], [[30.0]]])
    score_quantile_alpha_02 = torch.tensor([[[4.0]], [[40.0]]])
    expected = torch.stack(
        (
            _interval(pred, score_quantile_alpha_04),
            _interval(pred, score_quantile_alpha_02),
        ),
        dim=-1,
    )

    assert intervals.shape == (2, 2, 1, 1, 2, 2)
    torch.testing.assert_close(intervals, expected)


def test_calibrate_and_predict_end_to_end_on_multicell_data(calibrator):
    # Shape: (calibration_examples=10, time=3, spatial=2, channels=2).
    # For each (time, spatial, channel) cell, calibration residuals are
    # [1, ..., 10] plus that cell's offset. This makes score quantiles explicit:
    # alpha=0.4 selects 7 + offset, and alpha=0.2 selects 9 + offset.
    residual_ranks = torch.arange(1.0, 11.0).reshape(10, 1, 1, 1)
    cell_offsets = torch.arange(12.0).reshape(1, 3, 2, 2)
    expected_scores = residual_ranks + cell_offsets
    pred_calibration = torch.full_like(expected_scores, 100.0)
    true_calibration = pred_calibration + expected_scores

    calibrator.calibrate(true_calibration, pred_calibration)

    torch.testing.assert_close(calibrator.scores, expected_scores)

    score_quantile_alpha_04 = 7.0 + cell_offsets.squeeze(0)
    score_quantile_alpha_02 = 9.0 + cell_offsets.squeeze(0)
    torch.testing.assert_close(
        calibrator.score_quantile(alpha=0.4),
        score_quantile_alpha_04,
    )
    torch.testing.assert_close(
        calibrator.score_quantile(alpha=0.2),
        score_quantile_alpha_02,
    )

    pred_test = torch.arange(24.0).reshape(2, 3, 2, 2) + 200.0
    intervals = calibrator.predict(pred_test, alphas=[0.4, 0.2])

    expected_intervals = torch.stack(
        (
            _interval(pred_test, score_quantile_alpha_04),
            _interval(pred_test, score_quantile_alpha_02),
        ),
        dim=-1,
    )

    assert intervals.shape == (2, 3, 2, 2, 2, 2)
    torch.testing.assert_close(intervals, expected_intervals)


def test_calibrate_and_predict_supports_no_optional_dims():
    calibrator = AbsoluteErrorResidual()
    true = torch.tensor([[2.0, 5.0], [3.0, 7.0], [4.0, 9.0]])
    pred = torch.tensor([[1.0, 2.0], [1.0, 5.0], [1.0, 8.0]])

    calibrator.calibrate(true, pred)

    expected_scores = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    torch.testing.assert_close(calibrator.scores, expected_scores)
    torch.testing.assert_close(
        calibrator.score_quantile(alpha=0.5),
        torch.tensor([2.0, 2.0]),
    )

    pred_test = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    intervals = calibrator.predict(pred_test, alphas=0.5)
    expected = _interval(pred_test, torch.tensor([2.0, 2.0])).unsqueeze(-1)

    assert intervals.shape == (2, 2, 2, 1)
    torch.testing.assert_close(intervals, expected)


def test_calibrate_rejects_mismatched_shapes(calibrator):
    with pytest.raises(BeartypeCallHintParamViolation):
        calibrator.calibrate(torch.ones(2, 2), torch.ones(2))

    with pytest.raises(ValueError, match="same shape"):
        calibrator.calibrate(torch.ones(2, 2, 1, 1), torch.ones(3, 2, 1, 1))


def test_predict_rejects_uncalibrated_or_mismatched_inputs(calibrator):
    with pytest.raises(RuntimeError, match="must be calibrated"):
        calibrator.predict(torch.ones(1, 2, 1, 1), alphas=0.2)

    calibrator.cache_scores(torch.ones(4, 2, 1, 1))

    with pytest.raises(ValueError, match="trailing shape"):
        calibrator.predict(torch.ones(1, 3, 1, 1), alphas=0.2)


def test_score_quantile_rejects_invalid_alpha(calibrator):
    calibrator.cache_scores(torch.ones(4, 2, 1, 1))

    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrator.score_quantile(alpha=0.0)

    with pytest.raises(ValueError, match="too small"):
        calibrator.score_quantile(alpha=0.1)
