from math import prod

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


def test_score_quantile_uses_finite_sample_order_statistic_per_cell(
    calibrator,
    finite_sample_scores,
    finite_sample_score_quantiles,
):
    # Shape: (calibration_examples=4, time=2, spatial=1, channels=1). The score
    # quantile is checked independently for each time/spatial/channel cell. For n=4,
    # alpha=0.4 selects the 3rd smallest score and alpha=0.2 selects the 4th.
    calibrator.cache_scores(finite_sample_scores)

    torch.testing.assert_close(
        calibrator.score_quantile(alpha=0.4),
        finite_sample_score_quantiles[0.4],
    )
    torch.testing.assert_close(
        calibrator.score_quantile(alpha=0.2),
        finite_sample_score_quantiles[0.2],
    )


def test_predict_returns_symmetric_intervals_for_each_alpha(
    calibrator,
    pred,
    finite_sample_scores,
    finite_sample_score_quantiles,
):
    # Shape: (calibration_examples=4, time=2, spatial=1, channels=1). This
    # mirrors the score quantile pattern, but includes the singleton spatial axis
    # implied by the configured dims. The output interval shape is
    # (batch, optional_dims..., channel, lower_upper, alphas).
    calibrator.cache_scores(finite_sample_scores)

    intervals = calibrator.predict(pred, alphas=[0.4, 0.2])

    expected = torch.stack(
        (
            _interval(pred, finite_sample_score_quantiles[0.4]),
            _interval(pred, finite_sample_score_quantiles[0.2]),
        ),
        dim=-1,
    )

    assert intervals.shape == (2, 2, 1, 1, 2, 2)
    torch.testing.assert_close(intervals, expected)


@pytest.mark.parametrize(
    ("calibrator", "optional_shape", "channels"),
    [
        pytest.param(AbsoluteErrorResidual(), (), 2, id="no_optional_dims"),
        pytest.param(AbsoluteErrorResidual(temporal_dim=1), (2,), 1, id="temporal"),
        pytest.param(AbsoluteErrorResidual(spatial_dims=(1,)), (2,), 1, id="spatial"),
        pytest.param(
            AbsoluteErrorResidual(temporal_dim=1, spatial_dims=(2,)),
            (3, 2),
            2,
            id="temporal_and_spatial",
        ),
        pytest.param(
            AbsoluteErrorResidual(temporal_dim=1, spatial_dims=(2, 3)),
            (2, 2, 3),
            1,
            id="temporal_and_two_spatial_dims",
        ),
    ],
)
def test_calibrate_and_predict_preserves_optional_structure(
    calibrator,
    optional_shape,
    channels,
):
    trailing_shape = (*optional_shape, channels)
    residuals = torch.arange(1.0, 4.0).reshape(3, *([1] * len(trailing_shape)))
    cell_offsets = torch.arange(prod(trailing_shape), dtype=torch.float32).reshape(
        1, *trailing_shape
    )
    pred = torch.full((3, *trailing_shape), 100.0)
    true = pred + residuals + cell_offsets

    calibrator.calibrate(true, pred)

    expected_scores = residuals + cell_offsets
    torch.testing.assert_close(calibrator.scores, expected_scores)

    score_quantile = 2.0 + cell_offsets.squeeze(0)
    torch.testing.assert_close(
        calibrator.score_quantile(alpha=0.5),
        score_quantile,
    )

    pred_test = torch.arange(2 * prod(trailing_shape), dtype=torch.float32).reshape(
        2, *trailing_shape
    )
    intervals = calibrator.predict(pred_test, alphas=0.5)
    expected = _interval(pred_test, score_quantile).unsqueeze(-1)

    assert intervals.shape == (2, *trailing_shape, 2, 1)
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
