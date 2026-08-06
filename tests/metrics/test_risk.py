import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from autouq.metrics import CoverageLoss


def test_coverage_loss_returns_per_example_miscoverage_fraction():
    true = torch.tensor(
        [
            [[0.0], [2.0]],
            [[-1.0], [3.0]],
        ]
    )
    intervals = torch.tensor(
        [
            [[[-1.0, 1.0]], [[2.5, 4.0]]],
            [[[-1.0, 0.0]], [[2.0, 3.0]]],
        ]
    )

    losses = CoverageLoss()(intervals, true)

    # Example 0 misses one of two cells. Example 1 includes both targets,
    # including targets exactly equal to an interval endpoint.
    torch.testing.assert_close(losses, torch.tensor([0.5, 0.0]))


def test_coverage_loss_rejects_non_interval_shape():
    with pytest.raises((ValueError, BeartypeCallHintParamViolation)):
        CoverageLoss()(torch.ones(2, 3, 1), torch.ones(2, 1))
