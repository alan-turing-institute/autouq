import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from autouq.calibrators.conformal.absolute_error_residual import AbsoluteErrorResidual


@pytest.fixture
def calibrator():
    return AbsoluteErrorResidual()


def test_streaming_calibration_matches_single_shot_calibrate(calibrator):
    true = torch.tensor([[1.0], [2.0], [5.0]])
    pred = torch.tensor([[0.0], [0.0], [1.0]])

    calibrator.reset_calibration()
    calibrator.update_calibration(true[:2], pred[:2])
    calibrator.update_calibration(true[2:], pred[2:])

    reference = AbsoluteErrorResidual()
    reference.calibrate(true, pred)

    torch.testing.assert_close(
        calibrator._calibration_scores(), reference._calibration_scores()
    )
    torch.testing.assert_close(
        calibrator.score_quantile(0.5), reference.score_quantile(0.5)
    )


def test_calibration_scores_concatenate_lazily(calibrator):
    calibrator.reset_calibration()
    calibrator.update_calibration(torch.tensor([[1.0]]), torch.tensor([[0.0]]))

    # Not materialized until something actually needs it.
    assert calibrator.scores is None

    calibrator.update_calibration(torch.tensor([[4.0]]), torch.tensor([[0.0]]))
    torch.testing.assert_close(
        calibrator._calibration_scores(), torch.tensor([[1.0], [4.0]])
    )
    # Materialized (and cached) as soon as something did need it.
    assert calibrator.scores is not None


def test_materialized_scores_do_not_retain_duplicate_pending_chunks(calibrator):
    calibrator.reset_calibration()
    calibrator.update_calibration(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    calibrator.update_calibration(torch.tensor([[4.0]]), torch.tensor([[0.0]]))

    calibrator._calibration_scores()

    # Once folded into self.scores, the individual chunks are no longer held
    # onto - otherwise they'd sit around doubling memory for no reason.
    assert calibrator._pending_score_chunks == []


def test_streaming_can_resume_after_materialization_without_losing_data(calibrator):
    calibrator.reset_calibration()
    calibrator.update_calibration(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    calibrator.update_calibration(torch.tensor([[4.0]]), torch.tensor([[0.0]]))

    # Materialize mid-stream (e.g. an intermediate predict()/score_quantile() call).
    torch.testing.assert_close(
        calibrator._calibration_scores(), torch.tensor([[1.0], [4.0]])
    )

    # Streaming resumes after that - the already-materialized scores must not
    # be dropped.
    calibrator.update_calibration(torch.tensor([[9.0]]), torch.tensor([[0.0]]))

    torch.testing.assert_close(
        calibrator._calibration_scores(), torch.tensor([[1.0], [4.0], [9.0]])
    )


def test_accumulate_score_appends_precomputed_chunks(calibrator):
    calibrator.reset_calibration()
    calibrator.accumulate_score(torch.tensor([[1.0], [2.0]]))
    calibrator.accumulate_score(torch.tensor([[4.0]]))

    torch.testing.assert_close(
        calibrator._calibration_scores(), torch.tensor([[1.0], [2.0], [4.0]])
    )


def test_reset_calibration_clears_pending_and_cached_scores(calibrator):
    calibrator.calibrate(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    calibrator.reset_calibration()

    with pytest.raises(RuntimeError, match="must be calibrated"):
        calibrator._calibration_scores()


def test_calibrate_after_streaming_overwrites_pending_chunks(calibrator):
    calibrator.update_calibration(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    calibrator.calibrate(torch.tensor([[9.0]]), torch.tensor([[0.0]]))

    torch.testing.assert_close(calibrator._calibration_scores(), torch.tensor([[9.0]]))


def test_update_calibration_validates_chunk_shape(calibrator):
    # A calibration dimension is part of the TensorBNC type hint itself, so a
    # scalar is rejected by beartype before it would reach the manual check.
    with pytest.raises(BeartypeCallHintParamViolation):
        calibrator.update_calibration(torch.tensor(1.0), torch.tensor(1.0))


def test_accumulate_score_validates_chunk_shape(calibrator):
    with pytest.raises(ValueError, match="at least one sample"):
        calibrator.accumulate_score(torch.empty(0, 1))
