import pytest
import torch

from autouq.calibrators import AbsoluteErrorResidual


@pytest.fixture
def calibrator():
    # Tests use channels-last tensors with one singleton spatial axis.
    return AbsoluteErrorResidual(spatial_dims=(1,))


@pytest.fixture
def y_pred():
    # Shape: (batch=2, time=2, spatial=1, channels=1). The large values make
    # interval offsets easy to inspect in conformal interval tests.
    return torch.tensor(
        [
            [[[100.0]], [[200.0]]],
            [[[300.0]], [[400.0]]],
        ]
    )


def test_calibrate_caches_absolute_error_scores(calibrator):
    # Shape: (calibration_examples=2, time=2, spatial=1, channels=1). The
    # singleton spatial axis keeps the test aligned with spatial_dims=(1).
    y_true = torch.tensor(
        [
            [[[1.0]], [[4.0]]],
            [[[-2.0]], [[8.0]]],
        ]
    )
    y_pred = torch.tensor(
        [
            [[[0.5]], [[5.0]]],
            [[[1.0]], [[6.0]]],
        ]
    )

    calibrator.calibrate(y_true, y_pred)

    expected_scores = torch.tensor(
        [
            [[[0.5]], [[1.0]]],
            [[[3.0]], [[2.0]]],
        ]
    )
    torch.testing.assert_close(calibrator.scores, expected_scores)


def test_q_hat_uses_finite_sample_order_statistic_per_cell(calibrator):
    # Shape: (calibration_examples=4, time=2, spatial=1, channels=1). q_hat is
    # checked independently for each time/spatial/channel cell. For n=4,
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
        calibrator.q_hat(alpha=0.4),
        torch.tensor([[[3.0]], [[30.0]]]),
    )
    torch.testing.assert_close(
        calibrator.q_hat(alpha=0.2),
        torch.tensor([[[4.0]], [[40.0]]]),
    )


def test_predict_returns_symmetric_intervals_for_each_alpha(
    calibrator,
    y_pred,
):
    # Shape: (calibration_examples=4, time=2, spatial=1, channels=1). This
    # mirrors the q_hat score pattern, but includes the singleton spatial axis
    # implied by spatial_dims=(1). The output interval shape is
    # (batch, time, spatial, channels, lower_upper, alphas).
    spatiotemporal_conformal_scores = torch.tensor(
        [
            [[[1.0]], [[10.0]]],
            [[[4.0]], [[40.0]]],
            [[[2.0]], [[20.0]]],
            [[[3.0]], [[30.0]]],
        ]
    )
    calibrator.cache_scores(spatiotemporal_conformal_scores)

    intervals = calibrator.predict(y_pred, alphas=[0.4, 0.2])

    q_hat_alpha_04 = torch.tensor([[[3.0]], [[30.0]]])
    q_hat_alpha_02 = torch.tensor([[[4.0]], [[40.0]]])
    interval_alpha_04 = torch.stack(
        (
            y_pred - q_hat_alpha_04,
            y_pred + q_hat_alpha_04,
        ),
        dim=-1,
    )
    interval_alpha_02 = torch.stack(
        (
            y_pred - q_hat_alpha_02,
            y_pred + q_hat_alpha_02,
        ),
        dim=-1,
    )
    expected = torch.stack((interval_alpha_04, interval_alpha_02), dim=-1)

    assert intervals.shape == (2, 2, 1, 1, 2, 2)
    torch.testing.assert_close(intervals, expected)


def test_calibrate_and_predict_end_to_end_on_multicell_data(calibrator):
    # Shape: (calibration_examples=10, time=3, spatial=2, channels=2).
    # For each (time, spatial, channel) cell, calibration residuals are
    # [1, ..., 10] plus that cell's offset. This makes q_hat explicit:
    # alpha=0.4 selects 7 + offset, and alpha=0.2 selects 9 + offset.
    residual_ranks = torch.arange(1.0, 11.0).reshape(10, 1, 1, 1)
    cell_offsets = torch.arange(12.0).reshape(1, 3, 2, 2)
    expected_scores = residual_ranks + cell_offsets
    y_pred_calibration = torch.full_like(expected_scores, 100.0)
    y_true_calibration = y_pred_calibration + expected_scores

    calibrator.calibrate(y_true_calibration, y_pred_calibration)

    torch.testing.assert_close(calibrator.scores, expected_scores)

    q_hat_alpha_04 = 7.0 + cell_offsets.squeeze(0)
    q_hat_alpha_02 = 9.0 + cell_offsets.squeeze(0)
    torch.testing.assert_close(calibrator.q_hat(alpha=0.4), q_hat_alpha_04)
    torch.testing.assert_close(calibrator.q_hat(alpha=0.2), q_hat_alpha_02)

    y_pred_test = torch.arange(24.0).reshape(2, 3, 2, 2) + 200.0
    intervals = calibrator.predict(y_pred_test, alphas=[0.4, 0.2])

    interval_alpha_04 = torch.stack(
        (y_pred_test - q_hat_alpha_04, y_pred_test + q_hat_alpha_04),
        dim=-1,
    )
    interval_alpha_02 = torch.stack(
        (y_pred_test - q_hat_alpha_02, y_pred_test + q_hat_alpha_02),
        dim=-1,
    )
    expected_intervals = torch.stack((interval_alpha_04, interval_alpha_02), dim=-1)

    assert intervals.shape == (2, 3, 2, 2, 2, 2)
    torch.testing.assert_close(intervals, expected_intervals)


def test_calibrate_rejects_mismatched_shapes(calibrator):
    with pytest.raises(ValueError, match="same shape"):
        calibrator.calibrate(torch.ones(2, 2, 1, 1), torch.ones(2, 2, 1))


def test_predict_rejects_uncalibrated_or_mismatched_inputs(calibrator):
    with pytest.raises(RuntimeError, match="must be calibrated"):
        calibrator.predict(torch.ones(1, 2, 1, 1), alphas=0.2)

    calibrator.cache_scores(torch.ones(4, 2, 1, 1))

    with pytest.raises(ValueError, match="trailing shape"):
        calibrator.predict(torch.ones(1, 3, 1, 1), alphas=0.2)


def test_q_hat_rejects_invalid_alpha(calibrator):
    calibrator.cache_scores(torch.ones(4, 2, 1, 1))

    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrator.q_hat(alpha=0.0)

    with pytest.raises(ValueError, match="too small"):
        calibrator.q_hat(alpha=0.1)
