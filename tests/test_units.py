import pytest

from ahu_oa import units


def test_f_to_c_known_points():
    assert units.f_to_c(32.0) == pytest.approx(0.0)
    assert units.f_to_c(212.0) == pytest.approx(100.0)


def test_round_trip_temperature():
    assert units.c_to_f(units.f_to_c(55.0)) == pytest.approx(55.0)


def test_delta_has_no_offset():
    # A 3.6 degF difference is 2.0 degC (the dataset quotes biases in degC)
    assert units.delta_f_to_delta_c(3.6) == pytest.approx(2.0)


def test_cfm_to_ls():
    assert units.cfm_to_ls(1.0) == pytest.approx(0.47195, abs=1e-5)
    assert units.ls_to_cfm(units.cfm_to_ls(1000.0)) == pytest.approx(1000.0)


def test_pressure_and_area():
    assert units.inwg_to_pa(1.6) == pytest.approx(398.5, abs=0.1)
    assert units.ft2_to_m2(38353.10) == pytest.approx(3563.1, abs=0.5)
