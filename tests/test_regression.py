"""Tests for the linearised regression of the outdoor air fraction on the damper."""
import numpy as np
import pandas as pd
import pytest

from ahu_oa import regression


def synth(n=2000, a=0.1, b=0.5, c=0.0, d=0.0, fan_const=None, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    oa = rng.uniform(0, 35, n)
    ra = np.full(n, 22.0)
    damper = rng.uniform(0, 1, n)
    fan = np.full(n, fan_const) if fan_const is not None else rng.uniform(0.4, 0.9, n)
    f = a + b * damper + c * fan + d * damper * fan
    ma = ra + f * (oa - ra) + rng.normal(0, noise, n)
    df = pd.DataFrame({"OA_TEMP_C": oa, "RA_TEMP_C": ra, "MA_TEMP_C": ma,
                       "OA_DMPR_DM": damper, "SF_CS": fan})
    df["well_conditioned"] = (df["OA_TEMP_C"] - df["RA_TEMP_C"]).abs() >= 3.0
    return df


def test_recovers_a_and_b():
    m = regression.fit(synth())
    assert m.coefs["a"] == pytest.approx(0.1, abs=0.02)
    assert m.coefs["b"] == pytest.approx(0.5, abs=0.02)
    assert m.warnings() == []


def test_standard_errors_are_positive_and_small():
    m = regression.fit(synth())
    assert 0 < m.stderr["a"] < 0.01
    assert 0 < m.stderr["b"] < 0.01
    assert m.rmse_K == pytest.approx(0.05, abs=0.01)   # equals the injected noise


def test_recovers_fan_term():
    m = regression.fit(synth(a=0.05, b=0.5, c=0.2), use_fan=True)
    assert m.coefs["c"] == pytest.approx(0.2, abs=0.03)


def test_recovers_interaction_term():
    m = regression.fit(synth(a=0.05, b=0.3, c=0.1, d=0.4, n=6000), use_fan=True, interaction=True)
    assert m.coefs["d"] == pytest.approx(0.4, abs=0.06)


def test_constant_fan_is_detected_as_collinear():
    # A constant fan column is a multiple of the intercept column.
    m = regression.fit(synth(fan_const=0.6), use_fan=True)
    assert m.cond > 1e8
    assert any("ill-conditioned" in w for w in m.warnings())


def test_interaction_requires_fan():
    with pytest.raises(ValueError):
        regression.fit(synth(), interaction=True)


def test_prediction_works_on_low_delta_row():
    df = synth()
    m = regression.fit(df)
    low = df.iloc[:1].copy()
    low["OA_TEMP_C"] = low["RA_TEMP_C"]  # zero temperature difference
    assert np.isfinite(m.predict_fraction(low)).all()


def test_prediction_does_not_use_temperatures():
    df = synth()
    m = regression.fit(df)
    shifted = df.copy()
    shifted["OA_TEMP_C"] += 10.0
    np.testing.assert_allclose(m.predict_fraction(df), m.predict_fraction(shifted))


def test_prediction_matches_a_plus_b_damper():
    df = synth().head(10)
    m = regression.fit(synth())
    expected = m.coefs["a"] + m.coefs["b"] * df["OA_DMPR_DM"].to_numpy()
    np.testing.assert_allclose(m.predict_fraction(df), expected)


def test_alternative_damper_column_is_used():
    df = synth()
    df["OA_DMPR"] = df["OA_DMPR_DM"]
    df["OA_DMPR_DM"] = 0.5                      # command carries no information
    m = regression.fit(df, damper_col="OA_DMPR")
    assert m.damper_col == "OA_DMPR"
    assert m.coefs["b"] == pytest.approx(0.5, abs=0.02)


def test_well_conditioned_only_uses_fewer_rows():
    df = synth()
    all_m = regression.fit(df)
    flg = regression.fit(df, well_conditioned_only=True)
    assert flg.n_train < all_m.n_train
    assert flg.n_train == int(df["well_conditioned"].sum())


def test_well_conditioned_only_needs_flag_column():
    with pytest.raises(KeyError):
        regression.fit(synth().drop(columns="well_conditioned"), well_conditioned_only=True)


def test_missing_values_are_dropped_not_fatal():
    df = synth()
    df.loc[df.index[:10], "MA_TEMP_C"] = np.nan
    assert regression.fit(df).n_train == len(df) - 10


def test_missing_column_raises():
    with pytest.raises(KeyError):
        regression.fit(synth().drop(columns="OA_DMPR_DM"))


def test_too_few_rows_raises():
    with pytest.raises(ValueError):
        regression.fit(synth().head(2))


def test_negative_b_warns():
    m = regression.fit(synth(a=0.6, b=-0.3))
    assert any("not positive" in w for w in m.warnings())


def test_fully_open_damper_above_one_warns():
    m = regression.fit(synth(a=0.5, b=0.9))
    assert any("exceeds 1" in w for w in m.warnings())


def test_fan_signal_can_be_ignored_when_not_used():
    df = synth().drop(columns="SF_CS")
    assert regression.fit(df).coefs.keys() == {"a", "b"}
