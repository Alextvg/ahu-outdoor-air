import numpy as np
import pandas as pd

from ahu_oa import diagnostics, physics, regression
from test_regression import synth


def with_time(df):
    df = df.copy()
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="1h")
    return df


def test_fan_variation_detects_constant_signal():
    v = diagnostics.fan_variation(synth(fan_const=0.6))
    assert v["varies"] is False and v["n_unique"] == 1


def test_fan_variation_detects_varying_signal():
    assert diagnostics.fan_variation(synth())["varies"] is True


def test_compare_all_vs_flagged_has_two_rows():
    t = diagnostics.compare_all_vs_flagged(synth())
    assert list(t["fit"]) == ["all_rows", "well_conditioned_only"]
    assert t.loc[0, "n_train"] > t.loc[1, "n_train"]
    assert abs(t.loc[0, "b"] - t.loc[1, "b"]) < 0.03


def test_coefficient_stability_by_month_skips_small_groups():
    df = with_time(synth(n=24 * 70))
    t = diagnostics.coefficient_stability(df, by="month", min_rows=100)
    assert {"group", "a", "b"} <= set(t.columns)
    assert len(t) >= 2
    assert (t["n_train"] >= 100).all()


def test_coefficient_stability_by_half():
    t = diagnostics.coefficient_stability(with_time(synth()), by="half")
    assert set(t["group"]) == {"first", "second"}


def test_residual_by_damper_is_small_for_correct_model():
    df = synth(noise=0.02)
    df["f_phys"] = physics.oa_fraction_from_temps(
        df["MA_TEMP_C"], df["RA_TEMP_C"], df["OA_TEMP_C"], min_delta=3.0)
    m = regression.fit(df)
    t = diagnostics.residual_by_damper(df, m, bins=5)
    assert len(t) == 5
    assert t["mean_resid"].abs().max() < 0.02
