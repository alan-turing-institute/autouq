import math
from unittest.mock import patch
from weakref import ref

import pytest
import torch

from autouq.calibrators import (
    EnsembleRCPS,
    RCPSCalibrator,
    ScaledIntervalRCPS,
)


def _uncertainty_prediction(
    centre: torch.Tensor,
    lower_width: float | torch.Tensor = 1.0,
    upper_width: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    lower = torch.broadcast_to(torch.as_tensor(lower_width), centre.shape)
    upper = torch.broadcast_to(torch.as_tensor(upper_width), centre.shape)
    return torch.stack((centre, lower, upper), dim=-1)


@pytest.fixture
def calibration_data():
    # Every target needs lambda=1.25 to enter its symmetric interval. With 200
    # independent calibration examples, the Hoeffding margin at delta=0.05 is
    # below alpha=0.1, so this exact coverage threshold is feasible.
    true = torch.full((200, 1), 1.25)
    pred = _uncertainty_prediction(torch.zeros_like(true))
    return true, pred


def test_risk_upper_bound_is_empirical_risk_plus_hoeffding_margin(
    calibration_data,
):
    true, pred = calibration_data
    calibrator = ScaledIntervalRCPS(alphas=0.1, deltas=0.05)
    calibrator.calibrate(true, pred)

    margin = math.sqrt(math.log(1 / 0.05) / (2 * true.shape[0]))

    assert calibrator.risk_upper_bound(
        lambda_=1.0, alpha=0.1, delta=0.05
    ) == pytest.approx(1.0 + margin)
    assert calibrator.risk_upper_bound(
        lambda_=1.25, alpha=0.1, delta=0.05
    ) == pytest.approx(margin)


def test_lambda_hat_finds_smallest_risk_controlling_interval_scale(
    calibration_data,
):
    true, pred = calibration_data
    calibrator = ScaledIntervalRCPS(alphas=0.1, deltas=0.05, search_tolerance=1e-6)
    calibrator.calibrate(true, pred)

    lambda_hat = calibrator.lambda_hat(alpha=0.1, delta=0.05)

    assert lambda_hat == pytest.approx(1.25, abs=calibrator.search_tolerance)
    assert calibrator.lambda_hats[(0.1, 0.05)] == lambda_hat


def test_predict_returns_alpha_delta_product_with_asymmetric_intervals(
    calibration_data,
):
    true, pred = calibration_data
    calibrator = ScaledIntervalRCPS(
        alphas=[0.1, 0.2], deltas=[0.05, 0.2], search_tolerance=1e-6
    )
    calibrator.calibrate(true, pred)

    centre = torch.tensor([[10.0], [20.0]])
    pred_test = _uncertainty_prediction(centre, lower_width=2.0, upper_width=1.0)
    intervals = calibrator.predict(pred_test)

    expected_one = torch.stack(
        (
            centre - 1.25 * 2.0,
            centre + 1.25,
        ),
        dim=-1,
    )
    expected = expected_one.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 2, 2)

    assert intervals.shape == (2, 1, 2, 2, 2)
    torch.testing.assert_close(intervals, expected)
    assert set(calibrator.lambda_hats) == {
        (0.1, 0.05),
        (0.1, 0.2),
        (0.2, 0.05),
        (0.2, 0.2),
    }


def test_smaller_delta_fits_no_smaller_scale():
    true = torch.linspace(0, 2, 1_000).unsqueeze(-1)
    pred = _uncertainty_prediction(torch.zeros_like(true))
    calibrator = ScaledIntervalRCPS(
        alphas=0.2,
        deltas=[0.05, 0.2],
        search_tolerance=1e-6,
    )

    calibrator.calibrate(true, pred)

    assert calibrator.lambda_hat(0.2, 0.05) >= calibrator.lambda_hat(0.2, 0.2)


def test_calibrate_clears_scales_computed_for_previous_data(calibration_data):
    true, pred = calibration_data
    calibrator = ScaledIntervalRCPS(alphas=0.1, deltas=0.05)
    calibrator.calibrate(true, pred)
    assert calibrator.lambda_hat(alpha=0.1, delta=0.05) > 0

    calibrator.calibrate(torch.zeros_like(true), pred)

    assert calibrator.lambda_hats == {(0.1, 0.05): 0.0}


@pytest.mark.parametrize(
    ("n_calibration", "target", "width"),
    [
        pytest.param(10, 0.0, 1.0, id="bound_margin"),
        pytest.param(200, 1.0, 0.0, id="zero_width"),
    ],
)
def test_calibrate_rejects_infeasible_scale(n_calibration, target, width):
    true = torch.full((n_calibration, 1), target)
    pred = _uncertainty_prediction(torch.zeros_like(true), width, width)
    calibrator = ScaledIntervalRCPS(alphas=0.1, deltas=0.05, max_search_steps=4)

    with pytest.raises(RuntimeError, match="Could not find"):
        calibrator.calibrate(true, pred)


def test_hoeffding_bound_validates_bounded_loss_contract():
    class OutOfBoundsLoss:
        def __call__(self, prediction_set, true):
            del prediction_set
            return torch.full((true.shape[0],), 1.1)

    true = torch.zeros(200, 1)
    pred = _uncertainty_prediction(torch.zeros_like(true))

    calibrator = ScaledIntervalRCPS(alphas=0.1, deltas=0.05, loss=OutOfBoundsLoss())

    with pytest.raises(ValueError, match=r"HoeffdingBound.*\[0, 1\]"):
        calibrator.calibrate(true, pred)


def test_custom_bound_can_accept_unbounded_loss():
    class UnboundedLoss:
        def __call__(self, prediction_set, true):
            del prediction_set
            return torch.full((true.shape[0],), 2.0)

    class MaximumBound:
        def __call__(self, losses, delta):
            del delta
            return float(losses.max().item())

    true = torch.zeros(2, 1)
    pred = _uncertainty_prediction(torch.zeros_like(true))
    calibrator = ScaledIntervalRCPS(
        alphas=2.0,
        deltas=0.05,
        loss=UnboundedLoss(),
        bound=MaximumBound(),
    )
    calibrator.calibrate(true, pred)

    assert calibrator.risk_upper_bound(lambda_=0.0, alpha=2.0, delta=0.05) == 2.0
    assert calibrator.lambda_hat(alpha=2.0, delta=0.05) == 0.0


def test_predict_requires_calibration_and_matching_structure(calibration_data):
    calibrator = ScaledIntervalRCPS(alphas=0.1, deltas=0.05)
    with pytest.raises(RuntimeError, match="must be calibrated"):
        calibrator.predict(_uncertainty_prediction(torch.zeros(1, 1)))

    true, pred = calibration_data
    calibrator.calibrate(true, pred)
    with pytest.raises(ValueError, match="trailing shape"):
        calibrator.predict(_uncertainty_prediction(torch.zeros(1, 2)))


@pytest.mark.parametrize(
    ("alphas", "deltas", "match"),
    [
        pytest.param([], 0.05, "at least one risk level alpha", id="empty_alpha"),
        pytest.param(-0.1, 0.05, "non-negative", id="invalid_alpha"),
        pytest.param(0.1, [], "at least one delta", id="empty_delta"),
        pytest.param(0.1, 1.0, "open interval", id="invalid_delta"),
    ],
)
def test_constructor_validates_risk_and_error_levels(alphas, deltas, match):
    with pytest.raises(ValueError, match=match):
        ScaledIntervalRCPS(alphas=alphas, deltas=deltas)


def test_rcps_base_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        # Deliberately instantiate the abstract class to verify runtime rejection.
        RCPSCalibrator(alphas=0.1, deltas=0.05)  # ty: ignore[call-non-callable]


def test_ensemble_rcps_accepts_raw_ensemble_predictions():
    true = torch.full((200, 1), 1.25)
    calibration_ensemble = torch.tensor([-1.0, 1.0]).expand(200, 1, 2)
    calibrator = EnsembleRCPS(alphas=0.1, deltas=0.05, search_tolerance=1e-6)
    calibrator.calibrate(true, calibration_ensemble)

    lambda_hat = calibrator.lambda_hat(alpha=0.1, delta=0.05)
    test_ensemble = torch.tensor([[[8.0, 12.0]], [[18.0, 22.0]]])
    intervals = calibrator.predict(test_ensemble)

    assert lambda_hat == pytest.approx(1.25 / 0.9, abs=calibrator.search_tolerance)
    expected = torch.tensor(
        [
            [[[[7.5]], [[12.5]]]],
            [[[[17.5]], [[22.5]]]],
        ]
    )
    assert intervals.shape == (2, 1, 2, 1, 1)
    torch.testing.assert_close(intervals, expected)


def test_ensemble_quantile_mode_preserves_asymmetric_widths():
    calibration_ensemble = torch.tensor([0.0, 1.0, 2.0, 10.0]).expand(200, 1, 4)
    true = torch.full((200, 1), 8.8)
    calibrator = EnsembleRCPS(alphas=0.1, deltas=0.05, search_tolerance=1e-6)
    calibrator.calibrate(true, calibration_ensemble)

    intervals = calibrator.predict(calibration_ensemble[:1])

    assert calibrator.lambda_hat(alpha=0.1, delta=0.05) == pytest.approx(1.0)
    expected = torch.tensor([[[[[0.15]], [[8.8]]]]])
    torch.testing.assert_close(intervals, expected)


def test_ensemble_std_mode_uses_symmetric_population_scale():
    true = torch.full((200, 1), 1.25)
    calibration_ensemble = torch.tensor([-1.0, 1.0]).expand(200, 1, 2)
    calibrator = EnsembleRCPS(
        alphas=0.1,
        deltas=0.05,
        mode="std",
        search_tolerance=1e-6,
    )
    calibrator.calibrate(true, calibration_ensemble)

    assert calibrator.lambda_hat(alpha=0.1, delta=0.05) == pytest.approx(1.25)


@pytest.mark.parametrize("mode", ["quantile", "std"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cache_statistics", [True, False])
def test_ensemble_rcps_low_precision_constant_ensemble(mode, dtype, cache_statistics):
    true = torch.ones(200, 1, dtype=dtype)
    ensemble = torch.zeros(200, 1, 2, dtype=dtype)
    calibrator = EnsembleRCPS(
        alphas=0.1, deltas=0.05, mode=mode, cache_statistics=cache_statistics
    )

    calibrator.calibrate(true, ensemble)
    intervals = calibrator.predict(ensemble[:1])

    reference = EnsembleRCPS(alphas=0.1, deltas=0.05, mode=mode)
    reference.calibrate(true.float(), ensemble.float())
    assert calibrator.lambda_hats == reference.lambda_hats
    assert intervals.dtype == torch.float32
    torch.testing.assert_close(intervals, reference.predict(ensemble[:1].float()))
    torch.testing.assert_close(intervals.flatten(), torch.tensor([-1.0, 1.0]))


@pytest.mark.parametrize("cache_statistics", [True, False])
def test_ensemble_rcps_quantile_cache_preserves_results(cache_statistics):
    ensemble = torch.tensor([-1.0, 0.0, 1.0, 3.0]).expand(200, 2, 3, 1, 4)
    true = torch.linspace(0.0, 3.0, 1_200).reshape(200, 2, 3, 1)
    alphas = [0.1, 0.2]
    deltas = [0.05, 0.2]
    calibrator = EnsembleRCPS(
        alphas=alphas, deltas=deltas, cache_statistics=cache_statistics
    )

    with patch("torch.quantile", wraps=torch.quantile) as quantile:
        calibrator.calibrate(true, ensemble)
        if cache_statistics:
            assert quantile.call_count <= 2 * len(alphas)
        else:
            assert quantile.call_count > 2 * len(alphas)

    with patch("torch.quantile", wraps=torch.quantile) as quantile:
        intervals = calibrator.predict(ensemble[:2])
        if cache_statistics:
            assert quantile.call_count <= 2 * len(alphas)
        else:
            assert quantile.call_count > 2 * len(alphas)

    # Compare against the same interval family built independently, including
    # each alpha/delta output slice on a spatiotemporal tensor.
    centre = ensemble.mean(dim=-1)
    for index, alpha in enumerate(alphas):
        lower = torch.quantile(ensemble, alpha / 2, dim=-1)
        upper = torch.quantile(ensemble, 1 - alpha / 2, dim=-1)
        widths = _uncertainty_prediction(centre, centre - lower, upper - centre)
        reference = ScaledIntervalRCPS(alphas=alpha, deltas=deltas)
        reference.calibrate(true, widths)
        for delta in deltas:
            assert calibrator.lambda_hat(alpha, delta) == reference.lambda_hat(
                alpha, delta
            )
        torch.testing.assert_close(
            intervals[..., index, :], reference.predict(widths[:2])[..., 0, :]
        )


def test_ensemble_rcps_releases_previous_alpha_statistics(monkeypatch):
    ensemble = torch.tensor([-1.0, 0.0, 1.0, 3.0]).expand(200, 1, 4)
    true = torch.full((200, 1), 1.25)
    calibrator = EnsembleRCPS(alphas=[0.1, 0.2, 0.3], deltas=[0.05, 0.2])
    compute_statistics = calibrator._ensemble_statistics
    statistic_refs = {}

    def track_statistics(pred, alpha):
        statistics = compute_statistics(pred, alpha)
        # Weak references verify that previous alphas' tensors can be released,
        # without depending on the cache's container or retaining tensors here.
        for previous_alpha, references in statistic_refs.items():
            if previous_alpha != alpha:
                assert all(reference() is None for reference in references)
        statistic_refs[alpha] = [ref(tensor) for tensor in statistics]
        return statistics

    monkeypatch.setattr(calibrator, "_ensemble_statistics", track_statistics)
    calibrator.calibrate(true, ensemble)
    assert all(
        reference() is None
        for references in statistic_refs.values()
        for reference in references
    )
    calibrator.predict(ensemble[:2])
    assert all(
        reference() is None
        for references in statistic_refs.values()
        for reference in references
    )


@pytest.mark.parametrize("mode", ["quantile", "std"])
def test_ensemble_rcps_refreshes_statistics_between_calls(mode):
    true = torch.full((200, 1), 1.25, dtype=torch.float64)
    ensemble = torch.tensor([-1.0, 1.0], dtype=torch.float64).repeat(200, 1, 1)
    calibrator = EnsembleRCPS(alphas=0.1, deltas=0.05, mode=mode)
    calibrator.calibrate(true, ensemble)
    first_scale = calibrator.lambda_hat(0.1, 0.05)

    ensemble.mul_(2)
    calibrator.calibrate(true, ensemble)
    assert calibrator.lambda_hat(0.1, 0.05) == pytest.approx(
        first_scale / 2, abs=calibrator.search_tolerance
    )

    prediction = ensemble[:2].clone()
    intervals = calibrator.predict(prediction)
    prediction.add_(10)
    shifted_intervals = calibrator.predict(prediction)

    assert shifted_intervals.dtype == torch.float64
    torch.testing.assert_close(shifted_intervals, intervals + 10)


def test_ensemble_rcps_requires_calibrated_member_count():
    true = torch.full((200, 1), 1.25)
    calibration_ensemble = torch.tensor([-1.0, 1.0]).expand(200, 1, 2)
    calibrator = EnsembleRCPS(alphas=0.1, deltas=0.05)
    calibrator.calibrate(true, calibration_ensemble)

    test_ensemble = torch.tensor([[[-3.0, -1.0, 1.0, 3.0]]])
    with pytest.raises(ValueError, match="ensemble size"):
        calibrator.predict(test_ensemble)


def test_ensemble_rcps_validates_configuration_and_predictions():
    with pytest.raises(ValueError, match="min_scale"):
        EnsembleRCPS(alphas=0.1, deltas=0.05, min_scale=0.0)
    with pytest.raises(ValueError, match="min_scale"):
        EnsembleRCPS(alphas=0.1, deltas=0.05, min_scale=math.inf)
    with pytest.raises(ValueError, match="mode must"):
        EnsembleRCPS(alphas=0.1, deltas=0.05, mode="unknown")
    with pytest.raises(ValueError, match="quantile mode"):
        EnsembleRCPS(alphas=1.0, deltas=0.05)

    calibrator = EnsembleRCPS(alphas=0.1, deltas=0.05)
    with pytest.raises(ValueError, match="at least two members"):
        calibrator.calibrate(torch.ones(2, 1), torch.ones(2, 1, 1))
