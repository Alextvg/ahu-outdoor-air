"""Tests for the natural cubic spline estimator."""
import warnings

import numpy as np
import pandas as pd
import pytest

from ahu_oa import damper_model, spline_model, splits


def curved(n=6000, noise=0.01, seed=0, hours=True):
    """Fault-free-like data: a curved, monotone damper -> f characteristic."""
    rng = np.random.default_rng(seed)
    damper = rng.uniform(0, 1, n)
    f_true = 0.03 + 0.97 * damper ** 2.5                     # flat low, steep high
    f = f_true + rng.normal(0, noise, n)
    df = pd.DataFrame({"f_phys": f, "OA_DMPR": damper, "well_conditioned": True})
    if hours:
        df.index = pd.date_range("2020-01-01", periods=n, freq="1h")
    return df


def test_recovers_a_curved_relation():
    df = curved()
    est = spline_model.fit_spline(df, "OA_DMPR", n_knots=20)
    # Compare with the true inverse on the part where f is informative (f > 0.1)
    hi = df[df["f_phys"] > 0.1]
    err = np.abs(est.predict(hi["f_phys"]) - hi["OA_DMPR"]).mean()
    assert err < 0.03


def test_beats_linear_inversion_on_curved_data():
    df = curved()
    spl = spline_model.fit_spline(df, "OA_DMPR", n_knots=20)
    lin = damper_model.fit_direct(df, "OA_DMPR", degree=1)
    e_spl = np.abs(spl.predict(df["f_phys"]) - df["OA_DMPR"]).mean()
    e_lin = np.abs(lin.predict(df["f_phys"]) - df["OA_DMPR"]).mean()
    assert e_spl < e_lin / 2


def test_spline_passes_through_its_knots():
    # Interpolation property: at each knot fraction the prediction equals the knot value.
    est = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=12)
    np.testing.assert_allclose(est.predict(est.params["x"]), est.params["y"], atol=1e-9)
    assert np.all(np.diff(est.params["x"]) > 0)


def test_curve_is_smooth_between_knots():
    # Cubic pieces join with continuous slope: no jumps in the predicted curve.
    est = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=12)
    grid = np.linspace(est.params["x"][0], est.params["x"][-1], 4000)
    step = np.abs(np.diff(est.predict(grid)))
    assert step.max() < 0.05


def test_overshoot_between_knots_is_clipped():
    # A sharp step makes a cubic spline overshoot; the output must stay in the knot range.
    est = spline_model.SplineEstimator(
        "x", "spline", {"x": [0.0, 0.2, 0.4, 0.6, 0.8], "y": [0.0, 0.0, 1.0, 1.0, 1.0]}, 5)
    pred = est.predict(np.linspace(0.0, 0.8, 400))
    assert pred.min() >= 0.0 and pred.max() <= 1.0


def test_output_stays_within_training_range():
    df = curved()
    est = spline_model.fit_spline(df, "OA_DMPR", n_knots=20)
    pred = est.predict(np.linspace(-1.0, 2.0, 500))
    assert pred.min() >= df["OA_DMPR"].min() - 1e-9
    assert pred.max() <= df["OA_DMPR"].max() + 1e-9


def test_flat_outside_the_training_fractions():
    est = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=20)
    x = est.params["x"]
    lo, hi = est.predict([x[0] - 5.0, x[0] - 0.5]), est.predict([x[-1] + 0.5, x[-1] + 5.0])
    assert lo[0] == pytest.approx(lo[1]) and hi[0] == pytest.approx(hi[1])


def test_nan_fraction_gives_nan_position():
    est = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=10)
    out = est.predict([np.nan, 0.5])
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_predict_keeps_shape_for_scalar_and_2d_input():
    est = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=10)
    assert est.predict(0.5).shape == ()
    assert est.predict(np.full((2, 3), 0.5)).shape == (2, 3)


def test_only_well_conditioned_rows_are_used():
    df = curved()
    df.loc[df.index[:1000], "well_conditioned"] = False
    df.loc[df.index[:1000], "OA_DMPR"] = 0.0                  # poison the excluded rows
    est = spline_model.fit_spline(df, "OA_DMPR", n_knots=10)
    assert est.n_train == len(df) - 1000


def test_uniform_rule_places_knots_across_the_range():
    df = curved()
    q = spline_model.fit_spline(df, "OA_DMPR", n_knots=10, rule="quantile")
    u = spline_model.fit_spline(df, "OA_DMPR", n_knots=10, rule="uniform")
    assert np.ptp(np.diff(u.params["x"])) < np.ptp(np.diff(q.params["x"]))
    assert u.kind == "spline" and u.params["rule"] == "uniform"
    assert u.name.startswith("cubic_spline")


def test_matches_damper_estimator_interface():
    est = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=10)
    for attr in ("name", "kind", "params", "n_train", "predict"):
        assert hasattr(est, attr)


def test_invalid_inputs_raise():
    df = curved()
    with pytest.raises(ValueError):
        spline_model.fit_spline(df, "OA_DMPR", n_knots=2)
    with pytest.raises(ValueError):
        spline_model.fit_spline(df, "OA_DMPR", rule="bogus")
    with pytest.raises(ValueError):
        spline_model.fit_spline(df.head(20), "OA_DMPR", n_knots=10)
    with pytest.raises(KeyError):
        spline_model.fit_spline(df.drop(columns="OA_DMPR"), "OA_DMPR")


def test_constant_fraction_cannot_be_fitted():
    df = curved()
    df["f_phys"] = 0.5
    with pytest.raises(ValueError):
        spline_model.fit_spline(df, "OA_DMPR", n_knots=10)


def test_select_n_knots_returns_a_candidate_and_table():
    df = curved(n=8000)
    train, val = df.iloc[:6000], df.iloc[6000:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best, table = spline_model.select_n_knots(train, val, "OA_DMPR", candidates=(3, 8, 20))
    assert best in (3, 8, 20)
    assert list(table["n_knots"]) == [3, 8, 20]
    assert (table["n_scored"] > 0).all()
    assert best != 3                                  # 3 knots cannot follow the curve
    assert bool(table.loc[table["n_knots"] == best, "within_tolerance"].iloc[0])


def test_tolerance_picks_the_simplest_curve_that_is_nearly_as_good():
    df = curved(n=8000)
    train, val = df.iloc[:6000], df.iloc[6000:]
    cands = (10, 20, 60, 120, 250)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lowest, table = spline_model.select_n_knots(train, val, "OA_DMPR", cands, tolerance=0.0,
                                                 abs_tolerance=0.0)
        loose, _ = spline_model.select_n_knots(train, val, "OA_DMPR", cands, tolerance=0.10, abs_tolerance=0.0)
        anything, _ = spline_model.select_n_knots(train, val, "OA_DMPR", cands, tolerance=100.0, abs_tolerance=0.0)
    assert lowest == int(table.loc[table["val_mae"].idxmin(), "n_knots"])
    assert loose <= lowest
    # everything is within a huge tolerance, so the smallest local minimum wins
    minima = table.loc[spline_model.local_minima(table["val_mae"].to_numpy()), "n_knots"]
    assert anything == int(minima.min())


def test_negative_tolerance_raises():
    df = curved(n=8000)
    with pytest.raises(ValueError):
        spline_model.select_n_knots(df.iloc[:6000], df.iloc[6000:], "OA_DMPR", tolerance=-0.1)


def test_select_n_knots_uses_the_validation_rows_only():
    df = curved(n=8000)
    train, val = df.iloc[:6000].copy(), df.iloc[6000:].copy()
    val["well_conditioned"] = False                      # nothing scoreable
    with pytest.raises(ValueError):
        spline_model.select_n_knots(train, val, "OA_DMPR", candidates=(5, 10))


def test_select_n_knots_skips_candidates_that_cannot_be_fitted():
    df = curved(n=8000)
    train, val = df.iloc[:60], df.iloc[6000:]            # too few rows for 50 knots
    best, table = spline_model.select_n_knots(train, val, "OA_DMPR", candidates=(5, 50))
    assert best == 5 and np.isnan(table.loc[table["n_knots"] == 50, "val_mae"].iloc[0])


def test_default_knot_candidates_are_a_wide_increasing_grid():
    g = spline_model.DEFAULT_KNOT_CANDIDATES
    assert list(g) == sorted(set(g))
    assert g[0] <= 6 and g[-1] >= 200 and len(g) >= 8


def test_select_n_knots_warns_when_best_is_at_the_edge():
    df = curved(n=8000)
    train, val = df.iloc[:6000], df.iloc[6000:]
    # 3 and 4 knots cannot follow the curve, so the largest candidate wins and is at the edge
    with pytest.warns(UserWarning, match="edge"):
        spline_model.select_n_knots(train, val, "OA_DMPR", candidates=(3, 4, 6))


# ---- monotone variant (isotonic regression + PCHIP) ----------------------------------

def test_monotone_prediction_never_decreases():
    est = spline_model.fit_spline(curved(noise=0.08, seed=5), "OA_DMPR", n_knots=60, monotone=True)
    grid = np.linspace(-0.3, 1.4, 4000)
    assert np.all(np.diff(est.predict(grid)) >= -1e-12)


def test_monotone_knot_values_are_non_decreasing_even_for_noisy_bins():
    # Heavy noise makes raw bin medians wobble; isotonic regression must remove that.
    noisy = spline_model.fit_spline(curved(noise=0.15, seed=3), "OA_DMPR", n_knots=40)
    mono = spline_model.fit_spline(curved(noise=0.15, seed=3), "OA_DMPR", n_knots=40, monotone=True)
    assert np.any(np.diff(noisy.params["y"]) < 0)              # the cubic knots do wobble
    assert np.all(np.diff(mono.params["y"]) >= -1e-12)
    assert np.all(np.diff(mono.params["x"]) > 0)


def test_monotone_passes_through_its_isotonic_knots():
    est = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=15, monotone=True)
    np.testing.assert_allclose(est.predict(est.params["x"]), est.params["y"], atol=1e-9)


def test_monotone_does_not_overshoot_a_sharp_step():
    est = spline_model.SplineEstimator(
        "x", "spline", {"x": [0.0, 0.2, 0.4, 0.6, 0.8], "y": [0.0, 0.0, 1.0, 1.0, 1.0],
                        "monotone": True}, 5)
    pred = est.predict(np.linspace(0.0, 0.8, 400))
    assert pred.min() >= 0.0 and pred.max() <= 1.0
    assert np.all(np.diff(pred) >= -1e-12)


def test_monotone_recovers_a_curved_relation_and_beats_linear():
    df = curved()
    mono = spline_model.fit_spline(df, "OA_DMPR", n_knots=20, monotone=True)
    lin = damper_model.fit_direct(df, "OA_DMPR", degree=1)
    hi = df[df["f_phys"] > 0.1]
    assert np.abs(mono.predict(hi["f_phys"]) - hi["OA_DMPR"]).mean() < 0.03
    assert np.abs(mono.predict(df["f_phys"]) - df["OA_DMPR"]).mean() < \
        np.abs(lin.predict(df["f_phys"]) - df["OA_DMPR"]).mean() / 2


def test_monotone_flag_is_recorded_in_name_and_params():
    mono = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=10, monotone=True)
    cubic = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=10)
    assert mono.params["monotone"] is True and mono.name.startswith("monotone_spline")
    assert cubic.params["monotone"] is False and cubic.name.startswith("cubic_spline")


def test_monotone_keeps_nan_flat_extrapolation_and_range():
    df = curved()
    est = spline_model.fit_spline(df, "OA_DMPR", n_knots=20, monotone=True)
    assert np.isnan(est.predict([np.nan, 0.5])[0])
    x = est.params["x"]
    lo, hi = est.predict([x[0] - 5.0, x[0] - 0.5]), est.predict([x[-1] + 0.5, x[-1] + 5.0])
    assert lo[0] == pytest.approx(lo[1]) and hi[0] == pytest.approx(hi[1])
    assert est.predict(np.linspace(-1, 2, 300)).min() >= df["OA_DMPR"].min() - 1e-9


def test_old_estimators_without_the_monotone_key_still_predict():
    est = spline_model.SplineEstimator(
        "x", "spline", {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.5, 1.0]}, 3)   # no "monotone" key
    assert est.predict(0.25) == pytest.approx(0.25)


def test_select_n_knots_works_for_the_monotone_variant():
    df = curved(n=8000)
    train, val = df.iloc[:6000], df.iloc[6000:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best, table = spline_model.select_n_knots(train, val, "OA_DMPR", (3, 8, 20, 60), monotone=True)
    assert best in (3, 8, 20, 60) and best != 3
    assert table["val_mae"].notna().all()


# ---- local-minimum knot selection ----------------------------------------------------

def test_local_minima_flags_dips_and_ends():
    v = [5.0, 3.0, 4.0, 2.0, 2.5, 1.0]
    assert list(spline_model.local_minima(v)) == [False, True, False, True, False, True]
    assert list(spline_model.local_minima([1.0, 2.0, 3.0])) == [True, False, False]   # rising
    assert list(spline_model.local_minima([3.0, 2.0, 1.0])) == [False, False, True]   # falling


def test_local_minima_ignores_nan_and_handles_all_nan():
    v = [4.0, np.nan, 2.0, 3.0]
    assert list(spline_model.local_minima(v)) == [False, False, True, False]
    assert not spline_model.local_minima([np.nan, np.nan]).any()


def _patch_scores(monkeypatch, maes):
    """Make fit_spline/predict produce the given validation MAE for each candidate."""
    cands = list(maes)

    def fake_fit(train, col, n_knots, rule="quantile", monotone=False):
        return spline_model.SplineEstimator("x", "spline", {"mae": maes[n_knots]}, 1)

    monkeypatch.setattr(spline_model, "fit_spline", fake_fit)
    monkeypatch.setattr(spline_model.SplineEstimator, "predict",
                        lambda self, f: np.asarray(f, float) + self.params["mae"])
    val = pd.DataFrame({"f_phys": np.zeros(10), "OA_DMPR": np.zeros(10), "well_conditioned": True})
    return cands, val


def test_selects_smaller_local_minimum_when_nearly_as_good(monkeypatch):
    # Shape of the real curve: dip at 20, bump, then a deeper minimum at 120 / 250.
    maes = {10: 0.0019, 20: 0.00112, 30: 0.00127, 60: 0.0015, 120: 0.00066, 250: 0.00071}
    cands, val = _patch_scores(monkeypatch, maes)
    k, table = spline_model.select_n_knots(None, val, "OA_DMPR", cands, abs_tolerance=0.0005)
    assert k == 20
    assert table.loc[table["n_knots"] == 20, "eligible"].iloc[0]
    assert table.loc[table["n_knots"] == 120, "local_min"].iloc[0]


def test_zero_tolerances_give_the_plain_lowest_error(monkeypatch):
    maes = {10: 0.0019, 20: 0.00112, 30: 0.00127, 60: 0.0015, 120: 0.00066, 250: 0.00071}
    cands, val = _patch_scores(monkeypatch, maes)
    k, _ = spline_model.select_n_knots(None, val, "OA_DMPR", cands, tolerance=0.0, abs_tolerance=0.0)
    assert k == 120



def test_falling_curve_warns_that_the_dip_may_be_beyond_the_grid(monkeypatch):
    maes = {10: 0.003, 20: 0.002, 30: 0.001}
    cands, val = _patch_scores(monkeypatch, maes)
    with pytest.warns(UserWarning, match="edge"):
        spline_model.select_n_knots(None, val, "OA_DMPR", cands)


def test_chosen_is_a_local_minimum_within_tolerance_on_real_fits():
    df = curved(n=8000)
    train, val = df.iloc[:6000], df.iloc[6000:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k, table = spline_model.select_n_knots(train, val, "OA_DMPR", (6, 10, 20, 60, 120, 250))
    row = table.loc[table["n_knots"] == k].iloc[0]
    assert row["local_min"] and row["within_tolerance"] and row["eligible"]


def test_negative_abs_tolerance_raises():
    df = curved(n=8000)
    with pytest.raises(ValueError):
        spline_model.select_n_knots(df.iloc[:6000], df.iloc[6000:], "OA_DMPR", abs_tolerance=-1.0)


def test_fit_records_rows_and_has_no_guard_parameters():
    est = spline_model.fit_spline(curved(), "OA_DMPR", n_knots=20)
    assert len(est.params["rows"]) == len(est.params["x"])
    assert sum(est.params["rows"]) == est.n_train
    assert "min_bin_rows" not in est.params
    with pytest.raises(TypeError):
        spline_model.fit_spline(curved(), "OA_DMPR", n_knots=20, min_bin_rows=5)
