"""Tests for the mixing-equation calculation of the outdoor air fraction."""
import numpy as np
import pytest

from ahu_oa import physics


def mixed(f, t_out, t_ret):
    """Mixed air temperature for a known fraction (forward mixing equation)."""
    return f * t_out + (1 - f) * t_ret


def test_fraction_recovers_known_value():
    # Verification on synthetic data: build mixed air from a known 30% fraction
    t_out, t_ret, f_true = 5.0, 22.0, 0.30
    est = physics.oa_fraction_from_temps(mixed(f_true, t_out, t_ret), t_ret, t_out)
    assert est == pytest.approx(f_true)


def test_fraction_vectorised():
    t_out = np.array([5.0, 10.0])
    t_ret = np.array([22.0, 22.0])
    f = np.array([0.1, 0.5])
    est = physics.oa_fraction_from_temps(mixed(f, t_out, t_ret), t_ret, t_out)
    assert est == pytest.approx(f)


def test_fraction_works_for_warm_outdoor_air():
    # Summer: outdoor warmer than return, so the temperature difference is positive.
    t_out, t_ret, f_true = 32.0, 22.0, 0.6
    est = physics.oa_fraction_from_temps(mixed(f_true, t_out, t_ret), t_ret, t_out)
    assert est == pytest.approx(f_true)


def test_ill_conditioned_returns_nan():
    # Outdoor and return within min_delta: the estimate is not trusted
    result = physics.oa_fraction_from_temps(21.0, 22.0, 21.5, min_delta=3.0)
    assert np.isnan(result)


def test_threshold_is_inclusive():
    # |T_out - T_ret| exactly equal to min_delta is accepted
    t_out, t_ret = 25.0, 22.0
    est = physics.oa_fraction_from_temps(mixed(0.5, t_out, t_ret), t_ret, t_out, min_delta=3.0)
    assert est == pytest.approx(0.5)
    just_below = physics.oa_fraction_from_temps(
        mixed(0.5, t_out, t_ret), t_ret, t_out, min_delta=3.0001)
    assert np.isnan(just_below)


def test_lower_threshold_accepts_more_rows():
    t_out = np.array([23.0, 24.5, 30.0])
    t_ret = np.full(3, 22.0)
    t_mix = mixed(0.4, t_out, t_ret)
    strict = physics.oa_fraction_from_temps(t_mix, t_ret, t_out, min_delta=3.0)
    loose = physics.oa_fraction_from_temps(t_mix, t_ret, t_out, min_delta=1.0)
    assert np.isnan(strict).sum() == 2
    assert np.isnan(loose).sum() == 0


def test_missing_temperature_gives_nan():
    assert np.isnan(physics.oa_fraction_from_temps(np.nan, 22.0, 5.0))


def test_fraction_is_not_clipped():
    # A mixed air reading colder than outdoor air gives f > 1; it must stay visible.
    est = physics.oa_fraction_from_temps(3.0, 22.0, 5.0)
    assert est == pytest.approx(19.0 / 17.0)
    assert est > 1.0


def test_fraction_plausible_range_and_nan():
    f = np.array([-0.2, -0.04, 0.0, 0.5, 1.0, 1.04, 1.3, np.nan])
    expected = [False, True, True, True, True, True, False, False]
    assert physics.fraction_plausible(f, tol=0.05).tolist() == expected


def test_fraction_plausible_strict_tolerance():
    assert physics.fraction_plausible(-0.04, tol=0.0) == False  # noqa: E712
    assert physics.fraction_plausible(0.5, tol=0.0) == True     # noqa: E712
