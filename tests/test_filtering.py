"""Tests for the row filters (fan-on rows, complete rows, conditioning flag)."""
import numpy as np
import pandas as pd
import pytest

from ahu_oa import filtering

# Columns the damper-position work needs. Passed explicitly so these tests do not
# depend on the module-level default list.
REQ = ["OA_TEMP_C", "RA_TEMP_C", "MA_TEMP_C", "SF_SPD_DM"]


def make_df():
    return pd.DataFrame({
        "SF_SPD_DM": [0,    1,    1,    1,    0],
        "OA_TEMP_C": [5.0,  5.0,  5.0, 21.5, 5.0],
        "RA_TEMP_C": [22.0, 22.0, 22.0, 22.0, 22.0],
        "MA_TEMP_C": [20.0, 18.0, np.nan, 21.8, 18.0],
    })


def test_fan_status_mask():
    assert filtering.occupied_mask(make_df(), "fan_status").tolist() == [False, True, True, True, False]


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        filtering.occupied_mask(make_df(), "bogus")


def test_valid_mask_drops_missing():
    df = make_df()
    assert filtering.valid_mask(df, REQ).tolist() == [True, True, False, True, True]


def test_well_conditioned_threshold():
    mask = filtering.well_conditioned_mask(make_df(), min_delta=3.0)
    # row 3: |21.5 - 22.0| = 0.5 K, too close
    assert mask.tolist() == [True, True, True, False, True]


def test_well_conditioned_threshold_changes_with_min_delta():
    df = make_df()
    assert filtering.well_conditioned_mask(df, min_delta=0.4).all()
    assert not filtering.well_conditioned_mask(df, min_delta=0.6).all()


def test_apply_filters_counts_and_flag():
    out, report = filtering.apply_filters(make_df(), required_cols=REQ)
    # fan-on rows 1-3, minus row 2 (NaN) = 2 rows kept
    assert len(out) == 2
    assert report["n_input"] == 5
    assert report["n_occupied"] == 3
    assert report["n_complete"] == 2
    # row 3 is kept but flagged as ill-conditioned
    assert report["n_well_conditioned"] == 1
    assert out["well_conditioned"].tolist() == [True, False]
    assert report["coverage_well_conditioned"] == pytest.approx(0.5)


def test_apply_filters_reports_settings():
    _, report = filtering.apply_filters(make_df(), required_cols=REQ, min_delta=2.0)
    assert report["min_delta"] == 2.0
    assert report["occupied_rule"] == "fan_status"


def test_apply_filters_empty_result_has_nan_coverage():
    df = make_df()
    df["SF_SPD_DM"] = 0
    out, report = filtering.apply_filters(df, required_cols=REQ)
    assert out.empty
    assert np.isnan(report["coverage_well_conditioned"])


def test_apply_filters_does_not_change_input():
    df = make_df()
    before = df.copy()
    filtering.apply_filters(df, required_cols=REQ)
    pd.testing.assert_frame_equal(df, before)
