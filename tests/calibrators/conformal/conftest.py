import pytest
import torch


@pytest.fixture
def finite_sample_scores():
    return torch.tensor(
        [
            [[[1.0]], [[10.0]]],
            [[[4.0]], [[40.0]]],
            [[[2.0]], [[20.0]]],
            [[[3.0]], [[30.0]]],
        ]
    )


@pytest.fixture
def finite_sample_score_quantiles():
    return {
        0.4: torch.tensor([[[3.0]], [[30.0]]]),
        0.2: torch.tensor([[[4.0]], [[40.0]]]),
    }
