import numpy as np
import pytest

from ahu_oa import physics


def test_fraction_recovers_known_value():
    # Verification on synthetic data: build mixed air from a known 30% fraction
    t_out, t_ret, f_true = 5.0, 22.0, 0.30
    t_mix = f_true * t_out + (1 - f_true) * t_ret
    assert physics.oa_fraction_from_temps(t_mix, t_ret, t_out) == pytest.approx(f_true)


def test_fraction_vectorised():
    t_out = np.array([5.0, 10.0])
    t_ret = np.array([22.0, 22.0])
    f = np.array([0.1, 0.5])
    t_mix = f * t_out + (1 - f) * t_ret
    assert physics.oa_fraction_from_temps(t_mix, t_ret, t_out) == pytest.approx(f)


def test_ill_conditioned_returns_nan():
    # Outdoor and return within min_delta: the estimate is not trusted
    result = physics.oa_fraction_from_temps(21.0, 22.0, 21.5, min_delta=3.0)
    assert np.isnan(result)


def test_flow_from_fraction():
    assert physics.oa_flow_from_fraction(0.25, 4000.0) == pytest.approx(1000.0)


def test_breathing_zone_flow_hand_calculation():
    # 100 m^2, 10 people: 2.5*10 + 0.3*100 = 55 L/s
    assert physics.breathing_zone_flow_ls(100.0, 10.0) == pytest.approx(55.0)


def test_floor_requirement_is_in_expected_range():
    # Dataset floor: about 3563 m^2 with the default office density.
    area = 3563.12
    people = physics.default_occupancy(area)
    vbz = physics.breathing_zone_flow_ls(area, people)
    assert 1500 < vbz < 1600  # roughly 1.5 m^3/s (lower bound)
