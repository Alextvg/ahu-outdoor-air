"""Tests for the error metrics, checked against hand calculations."""
import numpy as np
import pytest

from ahu_oa import metrics


def test_error_summary_hand_calculation():
    # errors (pred - true) = [0.5, 0, -1]
    s = metrics.error_summary([1.0, 2.0, 3.0], [1.5, 2.0, 2.0])
    assert s["n_total"] == 3 and s["n_scored"] == 3 and s["coverage"] == 1.0
    assert s["mae"] == pytest.approx(0.5)
    assert s["rmse"] == pytest.approx(np.sqrt(1.25 / 3))
    assert s["bias"] == pytest.approx(-0.5 / 3)
    assert s["nmae_pct"] == pytest.approx(25.0)       # 0.5 / mean(true)=2.0


def test_sign_convention_positive_bias_means_too_high():
    assert metrics.error_summary([0.2, 0.2], [0.3, 0.3])["bias"] == pytest.approx(0.1)


def test_perfect_estimate_has_zero_error():
    s = metrics.error_summary([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    assert s["mae"] == 0 and s["rmse"] == 0 and s["bias"] == 0


def test_nan_rows_are_left_out_and_coverage_reports_it():
    s = metrics.error_summary([1.0, 2.0, 3.0, 4.0], [1.0, np.nan, 3.0, 5.0])
    assert s["n_scored"] == 3
    assert s["coverage"] == pytest.approx(0.75)
    assert s["mae"] == pytest.approx(1.0 / 3)


def test_nan_in_truth_is_also_left_out():
    s = metrics.error_summary([np.nan, 2.0], [1.0, 2.5])
    assert s["n_scored"] == 1 and s["mae"] == pytest.approx(0.5)


def test_nothing_scored_gives_nan_not_an_error():
    s = metrics.error_summary([1.0, 2.0], [np.nan, np.nan])
    assert s["n_scored"] == 0 and s["coverage"] == 0
    assert np.isnan(s["mae"]) and np.isnan(s["rmse"]) and np.isnan(s["bias"])


def test_empty_input_does_not_divide_by_zero():
    s = metrics.error_summary([], [])
    assert s["n_total"] == 0 and np.isnan(s["coverage"])


def test_zero_mean_truth_gives_nan_nmae():
    assert np.isnan(metrics.error_summary([0.0, 0.0], [0.1, 0.1])["nmae_pct"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.error_summary([1.0, 2.0], [1.0])


def test_summarise_by_conditioning_splits_groups():
    truth = np.array([0.5, 0.5, 0.5, 0.5])
    pred = np.array([0.5, 0.6, 0.9, np.nan])
    wc = np.array([True, True, False, False])
    out = metrics.summarise_by_conditioning(truth, pred, wc)
    assert set(out) == {"all", "well_conditioned", "ill_conditioned"}
    assert out["all"]["n_total"] == 4
    assert out["well_conditioned"]["mae"] == pytest.approx(0.05)
    assert out["ill_conditioned"]["n_scored"] == 1
    assert out["ill_conditioned"]["coverage"] == pytest.approx(0.5)
    assert out["ill_conditioned"]["mae"] == pytest.approx(0.4)
