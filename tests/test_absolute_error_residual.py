import pytest
import torch

from autouq.calibrators import AbsoluteErrorResidual


def test_calibrate_caches_absolute_error_scores():
    calibrator = AbsoluteErrorResidual(spatial_dims=(1,))
    y_true = torch.tensor(
        [
            [[1.0], [4.0]],
            [[-2.0], [8.0]],
        ]
    )
    y_pred = torch.tensor(
        [
            [[0.5], [5.0]],
            [[1.0], [6.0]],
        ]
    )

    calibrator.calibrate(y_true, y_pred)

    expected_scores = torch.tensor(
        [
            [[0.5], [1.0]],
            [[3.0], [2.0]],
        ]
    )
    torch.testing.assert_close(calibrator.scores, expected_scores)


def test_q_hat_uses_finite_sample_order_statistic_per_cell():
    calibrator = AbsoluteErrorResidual(spatial_dims=(1,))
    calibrator.cache_scores(
        torch.tensor(
            [
                [[1.0], [10.0]],
                [[4.0], [40.0]],
                [[2.0], [20.0]],
                [[3.0], [30.0]],
            ]
        )
    )

    torch.testing.assert_close(
        calibrator.q_hat(alpha=0.4),
        torch.tensor([[3.0], [30.0]]),
    )
    torch.testing.assert_close(
        calibrator.q_hat(alpha=0.2),
        torch.tensor([[4.0], [40.0]]),
    )


def test_predict_returns_symmetric_intervals_for_each_alpha():
    calibrator = AbsoluteErrorResidual(spatial_dims=(1,))
    calibrator.cache_scores(
        torch.tensor(
            [
                [[[1.0]], [[10.0]]],
                [[[4.0]], [[40.0]]],
                [[[2.0]], [[20.0]]],
                [[[3.0]], [[30.0]]],
            ]
        )
    )
    y_pred = torch.tensor(
        [
            [[[100.0]], [[200.0]]],
            [[[300.0]], [[400.0]]],
        ]
    )

    intervals = calibrator.predict(y_pred, alphas=[0.4, 0.2])

    q_hat_alpha_04 = torch.tensor([[[3.0]], [[30.0]]])
    q_hat_alpha_02 = torch.tensor([[[4.0]], [[40.0]]])
    interval_alpha_04 = torch.stack(
        (y_pred - q_hat_alpha_04, y_pred + q_hat_alpha_04),
        dim=-1,
    )
    interval_alpha_02 = torch.stack(
        (y_pred - q_hat_alpha_02, y_pred + q_hat_alpha_02),
        dim=-1,
    )
    expected = torch.stack((interval_alpha_04, interval_alpha_02), dim=-1)

    assert intervals.shape == (2, 2, 1, 1, 2, 2)
    torch.testing.assert_close(intervals, expected)


def test_calibrate_rejects_mismatched_shapes():
    calibrator = AbsoluteErrorResidual(spatial_dims=(1,))

    with pytest.raises(ValueError, match="same shape"):
        calibrator.calibrate(torch.ones(2, 2, 1), torch.ones(2, 1))


def test_predict_rejects_uncalibrated_or_mismatched_inputs():
    calibrator = AbsoluteErrorResidual(spatial_dims=(1,))

    with pytest.raises(RuntimeError, match="must be calibrated"):
        calibrator.predict(torch.ones(1, 2, 1, 1), alphas=0.2)

    calibrator.cache_scores(torch.ones(4, 2, 1, 1))

    with pytest.raises(ValueError, match="trailing shape"):
        calibrator.predict(torch.ones(1, 3, 1, 1), alphas=0.2)


def test_q_hat_rejects_invalid_alpha():
    calibrator = AbsoluteErrorResidual(spatial_dims=(1,))
    calibrator.cache_scores(torch.ones(4, 2, 1))

    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrator.q_hat(alpha=0.0)

    with pytest.raises(ValueError, match="too small"):
        calibrator.q_hat(alpha=0.1)
