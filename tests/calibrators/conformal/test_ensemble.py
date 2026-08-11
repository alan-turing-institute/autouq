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


# ---------------------------------------------------------------------------
# Online (streaming) interface
# ---------------------------------------------------------------------------


def test_quantile_mode_streaming_calibration_matches_whole_tensor(calibrator):
    # Same fixture data as test_quantile_mode_calibrate_caches_interval_scores.
    pred = torch.tensor(
        [
            [[[[0.0, 10.0, 20.0, 30.0, 40.0]]]],
            [[[[100.0, 110.0, 120.0, 130.0, 140.0]]]],
        ]
    )
    true = torch.tensor([[[[35.0]]], [[[105.0]]]])

    reference = Ensemble(
        temporal_dim=1, spatial_dims=(2,), mode="quantile", ensemble_alpha=0.5
    )
    reference.calibrate(true, pred)

    calibrator.reset_calibration()
    for i in range(true.shape[0]):
        # Keepdim slices (not plain indexing) so each example retains its own
        # size-1 calibration axis - the axis update_calibration/accumulate_score
        # concatenate across examples.
        example_true, example_pred = true[i : i + 1], pred[i : i + 1]
        calibrator.reset_online()
        for member_idx in range(example_pred.shape[-1]):
            calibrator.update_online(example_pred[..., member_idx], true=example_true)
        calibrator.accumulate_score(calibrator.finalize_online())

    torch.testing.assert_close(calibrator._calibration_scores(), reference.scores)
    torch.testing.assert_close(
        calibrator.score_quantile(0.5), reference.score_quantile(0.5)
    )


def test_std_mode_streaming_calibration_matches_whole_tensor(std_calibrator):
    # Same fixture data as test_std_mode_calibrate_caches_normalized_scores.
    pred = torch.tensor(
        [
            [[[[8.0, 12.0]]]],
            [[[[18.0, 22.0]]]],
        ]
    )
    true = torch.tensor([[[[13.0]]], [[[20.0]]]])

    reference = Ensemble(temporal_dim=1, spatial_dims=(2,), mode="std")
    reference.calibrate(true, pred)

    std_calibrator.reset_calibration()
    for i in range(true.shape[0]):
        example_true, example_pred = true[i : i + 1], pred[i : i + 1]
        std_calibrator.reset_online()
        for member_idx in range(example_pred.shape[-1]):
            std_calibrator.update_online(
                example_pred[..., member_idx], true=example_true
            )
        std_calibrator.accumulate_score(std_calibrator.finalize_online())

    torch.testing.assert_close(std_calibrator._calibration_scores(), reference.scores)


def test_quantile_mode_streaming_chunked_matches_unchunked():
    torch.manual_seed(0)
    time, spatial, channel, n_members = 2, 7, 3, 5
    pred = torch.randn(time, spatial, channel, n_members)
    true = pred.mean(dim=-1) + 0.1 * torch.randn(time, spatial, channel)

    reference = Ensemble(mode="quantile", ensemble_alpha=0.3)
    reference_score = reference._score(true, pred)

    streaming = Ensemble(mode="quantile", ensemble_alpha=0.3)
    streaming.reset_online()
    for member_idx in range(n_members):
        streaming.update_online(pred[..., member_idx], true=true)
    chunked_score = streaming.finalize_online(chunk_size=3, chunk_dim=-2)

    torch.testing.assert_close(chunked_score, reference_score)


def test_std_mode_streaming_is_fully_incremental_via_welford():
    torch.manual_seed(1)
    time, spatial, channel, n_members = 2, 3, 2, 6
    pred = torch.randn(time, spatial, channel, n_members)
    true = torch.randn(time, spatial, channel)

    reference = Ensemble(mode="std")
    reference_score = reference._score(true, pred)

    streaming = Ensemble(mode="std")
    streaming.reset_online()
    for member_idx in range(n_members):
        streaming.update_online(pred[..., member_idx], true=true)
        # No members retained at any point - only running Welford accumulators.
        assert (
            not hasattr(streaming, "_online_members") or streaming._online_members == []
        )
    streaming_score = streaming.finalize_online()

    torch.testing.assert_close(streaming_score, reference_score, atol=1e-5, rtol=1e-5)


def test_finalize_online_interval_matches_predict(calibrator, finite_sample_scores):
    calibrator.cache_scores(finite_sample_scores)
    base = torch.tensor(
        [
            [[[100.0]], [[200.0]]],
            [[[300.0]], [[400.0]]],
        ]
    )
    ensemble = torch.stack(
        [base, base + 10.0, base + 20.0, base + 30.0, base + 40.0], dim=-1
    )

    reference_intervals = calibrator.predict(ensemble, alphas=[0.4, 0.2])

    streaming_intervals = []
    for example in ensemble:
        calibrator.reset_online()
        for member_idx in range(example.shape[-1]):
            calibrator.update_online(example[..., member_idx])
        streaming_intervals.append(
            calibrator.finalize_online_interval(alphas=[0.4, 0.2])
        )

    torch.testing.assert_close(
        torch.stack(streaming_intervals, dim=0), reference_intervals
    )


def test_ensemble_online_rejects_too_few_members(calibrator, std_calibrator):
    calibrator.reset_online()
    calibrator.update_online(torch.ones(1, 1, 1), true=torch.ones(1, 1, 1))
    with pytest.raises(ValueError, match="at least 2 members"):
        calibrator.finalize_online()

    std_calibrator.reset_online()
    std_calibrator.update_online(torch.ones(1, 1, 1), true=torch.ones(1, 1, 1))
    with pytest.raises(ValueError, match="at least 2 members"):
        std_calibrator.finalize_online()


def test_finalize_online_requires_true(calibrator):
    calibrator.reset_online()
    calibrator.update_online(torch.ones(1, 1, 1))
    calibrator.update_online(torch.ones(1, 1, 1))
    with pytest.raises(ValueError, match="requires `true`"):
        calibrator.finalize_online()


def test_finalize_online_rejects_invalid_chunk_size(calibrator):
    calibrator.reset_online()
    calibrator.update_online(torch.ones(1, 1, 1), true=torch.ones(1, 1, 1))
    calibrator.update_online(torch.ones(1, 1, 1))
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        calibrator.finalize_online(chunk_size=0)
