import numpy as np
import pandas as pd
import pytest

from ahu_oa import stuck_flag


def series(values, start="2020-01-01", freq="1min"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float)


def test_disagreement_is_absolute_and_drops_nan():
    est = series([0.5, np.nan, 0.2])
    cmd = [0.4, 0.4, 0.5]
    g = stuck_flag.disagreement(est, cmd)
    assert list(g) == pytest.approx([0.1, 0.3]) and len(g) == 2


def test_large_steady_gap_is_flagged_once_enough_rows_exist():
    gap = series([0.5] * 30)
    out = stuck_flag.rolling_flag(gap, threshold=0.1, window="60min", min_rows=10)
    assert not out["decided"].iloc[:9].any()          # fewer than 10 rows so far
    assert out["decided"].iloc[9:].all() and out["flag"].iloc[9:].all()
    assert not out["flag"].iloc[:9].any()


def test_small_gap_is_not_flagged():
    out = stuck_flag.rolling_flag(series([0.02] * 30), 0.1, "60min", 5)
    assert out["decided"].any() and not out["flag"].any()


def test_median_ignores_a_short_burst():
    gap = series([0.01] * 40 + [0.9] * 3 + [0.01] * 40)   # three bad rows in a long run
    out = stuck_flag.rolling_flag(gap, 0.1, "60min", 10)
    assert not out["flag"].any()


def test_window_forgets_old_rows():
    gap = series([0.5] * 20 + [0.0] * 60)
    out = stuck_flag.rolling_flag(gap, 0.1, "10min", 5)
    assert out["flag"].iloc[15] and not out["flag"].iloc[-1]


def test_rolling_flag_validates_arguments():
    gap = series([0.1] * 10)
    with pytest.raises(ValueError):
        stuck_flag.rolling_flag(gap, -0.1)
    with pytest.raises(ValueError):
        stuck_flag.rolling_flag(gap, 0.1, min_rows=0)
    with pytest.raises(TypeError):
        stuck_flag.rolling_flag(pd.Series([0.1, 0.2]), 0.1)


def test_rates_row_and_day_level():
    # day 1 has a flagged stretch, day 2 none
    idx = pd.date_range("2020-01-01", periods=48, freq="1h")
    flags = pd.DataFrame({"rolling_median": 0.0, "decided": True, "flag": False}, index=idx)
    flags.loc[idx[:6], "flag"] = True
    r = stuck_flag.flag_rates(flags)
    assert r["n_decided"] == 48 and r["row_flag_rate"] == pytest.approx(6 / 48)
    assert r["n_days"] == 2 and r["day_flag_rate"] == pytest.approx(0.5)


def test_rates_ignore_undecided_rows_and_handle_none():
    flags = pd.DataFrame({"rolling_median": np.nan, "decided": False, "flag": False},
                         index=pd.date_range("2020-01-01", periods=3, freq="1h"))
    r = stuck_flag.flag_rates(flags)
    assert r["n_decided"] == 0 and np.isnan(r["row_flag_rate"]) and r["n_days"] == 0


def test_choose_threshold_takes_the_smallest_that_meets_the_target():
    gap = series([0.04] * 200 + [0.15] * 200)       # half the rows have a 0.15 gap
    args = (gap, [0.01, 0.05, 0.2], "10min", 3)
    assert stuck_flag.choose_threshold(*args, max_false_alarm=0.6) == 0.05   # flags about 50%
    assert stuck_flag.choose_threshold(*args, max_false_alarm=0.0) == 0.2    # flags nothing
    assert stuck_flag.choose_threshold(*args, max_false_alarm=1.0) == 0.01   # anything goes


def test_choose_threshold_warns_when_nothing_meets_the_target():
    gap = series([0.5] * 100)
    with pytest.warns(UserWarning):
        assert stuck_flag.choose_threshold(gap, [0.1, 0.2], "60min", 5, 0.0) == 0.2


def test_choose_threshold_rejects_bad_input():
    gap = series([0.1] * 20)
    with pytest.raises(ValueError):
        stuck_flag.choose_threshold(gap, [], "60min", 5, 0.01)
    with pytest.raises(ValueError):
        stuck_flag.choose_threshold(gap, [0.1], "60min", 5, 1.5)
