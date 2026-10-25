import numpy as np
import pandas as pd
import pytest

from ahu_oa import damper_model, physics
from test_regression import synth


def with_fraction(df):
    df = df.copy()
    df["f_phys"] = physics.oa_fraction_from_temps(
        df["MA_TEMP_C"], df["RA_TEMP_C"], df["OA_TEMP_C"], min_delta=3.0)
    return df


def test_inverted_recovers_damper_position():
    df = with_fraction(synth(noise=0.02))
    est = damper_model.fit_inverted(df, "OA_DMPR_DM")
    wc = df[df["well_conditioned"]]
    err = est.predict(wc["f_phys"]) - wc["OA_DMPR_DM"]
    assert np.abs(err).mean() < 0.02
    assert est.params["b"] == pytest.approx(0.5, abs=0.02)


def test_direct_degree_one_recovers_damper_position():
    df = with_fraction(synth(noise=0.02))
    est = damper_model.fit_direct(df, "OA_DMPR_DM", degree=1)
    wc = df[df["well_conditioned"]]
    assert np.abs(est.predict(wc["f_phys"]) - wc["OA_DMPR_DM"]).mean() < 0.02


def test_direct_quadratic_beats_linear_inversion_on_curved_data():
    rng = np.random.default_rng(0)
    n = 4000
    damper = rng.uniform(0, 1, n)
    f = 0.05 + 0.2 * damper + 0.7 * damper ** 2          # curved damper characteristic
    oa = rng.uniform(0, 35, n)
    df = pd.DataFrame({"OA_TEMP_C": oa, "RA_TEMP_C": 22.0,
                       "MA_TEMP_C": 22.0 + f * (oa - 22.0) + rng.normal(0, 0.02, n),
                       "OA_DMPR": damper})
    df["well_conditioned"] = (df["OA_TEMP_C"] - 22.0).abs() >= 3.0
    df = with_fraction(df)
    wc = df[df["well_conditioned"]]
    inv = damper_model.fit_inverted(df, "OA_DMPR")
    quad = damper_model.fit_direct(df, "OA_DMPR", degree=2)
    err_inv = np.abs(inv.predict(wc["f_phys"]) - wc["OA_DMPR"]).mean()
    err_quad = np.abs(quad.predict(wc["f_phys"]) - wc["OA_DMPR"]).mean()
    assert err_quad < err_inv / 2


def test_nan_fraction_gives_nan_position():
    df = with_fraction(synth())
    for est in (damper_model.fit_inverted(df, "OA_DMPR_DM"),
                damper_model.fit_direct(df, "OA_DMPR_DM", 2)):
        assert np.isnan(est.predict([np.nan, 0.3])[0])
        assert np.isfinite(est.predict([np.nan, 0.3])[1])


def test_predictions_are_not_clipped():
    est = damper_model.DamperEstimator("x", "inverted", {"a": 0.1, "b": 0.5}, 10)
    assert est.predict([0.9])[0] == pytest.approx(1.6)
    assert est.predict([0.0])[0] == pytest.approx(-0.2)


def test_non_positive_slope_gives_nan_not_a_wrong_answer():
    est = damper_model.DamperEstimator("x", "inverted", {"a": 0.5, "b": -0.2}, 10)
    assert np.isnan(est.predict([0.3])).all()


def test_direct_needs_enough_rows():
    df = with_fraction(synth()).head(3)
    with pytest.raises(ValueError):
        damper_model.fit_direct(df, "OA_DMPR_DM", degree=2)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        damper_model.DamperEstimator("x", "other", {}, 1).predict([0.1])
