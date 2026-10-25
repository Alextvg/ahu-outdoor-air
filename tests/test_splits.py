import numpy as np
import pandas as pd
import pytest

from ahu_oa import splits


def make_df(days=140):
    idx = pd.date_range("2020-01-01", periods=days * 24, freq="1h")
    return pd.DataFrame({"x": np.arange(len(idx))}, index=idx)


def test_blocked_parts_do_not_overlap_and_cover_everything():
    df = make_df()
    parts = splits.split_data(df, "blocked")
    idx = [set(p.index) for p in parts.values()]
    assert idx[0].isdisjoint(idx[1]) and idx[0].isdisjoint(idx[2]) and idx[1].isdisjoint(idx[2])
    assert sum(len(p) for p in parts.values()) == len(df)


def test_blocked_default_ratio_is_about_60_20_20():
    parts = splits.split_data(make_df(), "blocked")
    n = sum(len(p) for p in parts.values())
    assert len(parts["train"]) / n == pytest.approx(0.6, abs=0.05)


def test_blocked_keeps_whole_blocks_together():
    parts = splits.split_data(make_df(), "blocked", block="7D")
    t0 = make_df().index.min()
    for p in parts.values():
        ids = ((p.index - t0) // pd.Timedelta("7D")).unique()
        for i in ids:
            in_block = (((p.index - t0) // pd.Timedelta("7D")) == i).sum()
            assert in_block == 7 * 24


def test_blocked_spreads_every_part_across_the_timeline():
    parts = splits.split_data(make_df(), "blocked")
    span = make_df().index.max() - make_df().index.min()
    for p in parts.values():
        assert (p.index.max() - p.index.min()) > span * 0.6


def test_chronological_is_ordered_in_time():
    parts = splits.split_data(make_df(), "chronological")
    assert parts["train"].index.max() < parts["val"].index.min()
    assert parts["val"].index.max() < parts["test"].index.min()
    assert len(parts["train"]) == pytest.approx(0.6 * len(make_df()), abs=1)


def test_requires_datetime_index():
    with pytest.raises(TypeError):
        splits.split_data(make_df().reset_index(drop=True))


def test_requires_sorted_index():
    with pytest.raises(ValueError):
        splits.split_data(make_df().iloc[::-1])


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        splits.split_data(make_df(), "random")


def test_empty_part_raises():
    with pytest.raises(ValueError):
        splits.split_data(make_df(days=5), "blocked")  # one block only: all train


def test_bad_fractions_raise():
    with pytest.raises(ValueError):
        splits.split_data(make_df(), "chronological", fractions=(0.5, 0.2, 0.2))


def test_bad_pattern_label_raises():
    with pytest.raises(ValueError):
        splits.split_data(make_df(), "blocked", pattern=("train", "holdout"))
