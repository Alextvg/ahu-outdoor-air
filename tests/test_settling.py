import pandas as pd
import pytest

from ahu_oa import settling


def runs_frame():
    """Nine 1-minute rows with two fan runs: rows 1-4 and rows 6-8."""
    idx = pd.date_range("2020-01-01 08:00", periods=9, freq="1min")
    on = pd.Series([0, 1, 1, 1, 1, 0, 1, 1, 1], index=idx).astype(bool)
    return idx, on


def test_each_fan_run_is_timed_from_its_own_start():
    idx, on = runs_frame()
    mask = settling.settled_mask(idx, on, "2min")
    assert mask.tolist() == [False, False, False, True, True, False, False, False, True]


def test_zero_settle_keeps_every_fan_on_row():
    idx, on = runs_frame()
    assert settling.settled_mask(idx, on, "0min").tolist() == on.tolist()


def test_fan_off_rows_are_never_settled():
    idx, on = runs_frame()
    assert not settling.settled_mask(idx, on, "1min")[~on].any()


def test_requires_datetime_index():
    _, on = runs_frame()
    with pytest.raises(TypeError):
        settling.settled_mask(pd.RangeIndex(9), on)


def test_requires_sorted_index():
    idx, on = runs_frame()
    with pytest.raises(ValueError):
        settling.settled_mask(idx[::-1], on[::-1])
