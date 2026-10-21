import numpy as np
import pandas as pd
import pytest

from ahu_oa import filtering


def make_df():
    return pd.DataFrame({
        "SYS_CTL":    [0,    1,    1,    1,    1],
        "SF_SPD_DM":  [0,    1,    1,    1,    0],
        "OA_TEMP_C":  [5.0,  5.0,  5.0, 21.5, 5.0],
        "RA_TEMP_C":  [22.0, 22.0, 22.0, 22.0, 22.0],
        "MA_TEMP_C":  [20.0, 18.0, np.nan, 21.8, 18.0],
        "SF_CS":      [0.0,  0.5,  0.5,  0.5,  0.5],
        "OA_DMPR_DM": [0.0,  0.1,  0.1,  0.1,  0.1],
        "SA_Ls":      [0.0,  900., 900., 900., 900.],
        "OA_Ls":      [0.0,  90.,  90.,  90.,  90.],
    })


def test_occupied_rules_differ():
    df = make_df()
    assert filtering.occupied_mask(df, "sys_ctl").sum() == 4
    assert filtering.occupied_mask(df, "fan_status").sum() == 3


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        filtering.occupied_mask(make_df(), "bogus")


def test_valid_mask_drops_missing():
    df = make_df()
    assert filtering.valid_mask(df).tolist() == [True, True, False, True, True]


def test_well_conditioned_threshold():
    df = make_df()
    mask = filtering.well_conditioned_mask(df, min_delta=3.0)
    # row 3: |21.5 - 22.0| = 0.5 K, too close
    assert mask.tolist() == [True, True, True, False, True]


def test_apply_filters_counts_and_flag():
    out, report = filtering.apply_filters(make_df())
    # occupied rows 1-4, minus row 2 (NaN) = 3 rows kept
    assert len(out) == 3
    assert report["n_input"] == 5
    assert report["n_occupied"] == 4
    assert report["n_complete"] == 3
    # row 3 is kept but flagged as ill-conditioned
    assert report["n_well_conditioned"] == 2
    assert out["well_conditioned"].tolist() == [True, False, True]


def test_apply_filters_does_not_change_input():
    df = make_df()
    before = df.copy()
    filtering.apply_filters(df)
    pd.testing.assert_frame_equal(df, before)
