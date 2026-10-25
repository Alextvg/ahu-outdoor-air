import pandas as pd
import pytest

from ahu_oa import loading


def make_df():
    return pd.DataFrame({
        "OA_TEMP": [32.0, 212.0],
        "OA_CFM": [100.0, 200.0],
        "SA_SP": [1.0, 1.6],
        "SF_CS": [0.0, 0.5],
        "SF_SPD": [0.9, 0.9],       # constant column
        "CHWC_VLV": [0.2, 1.5],     # out of 0-1 range
        "SYS_CTL": [0, 1],
    })


def test_convert_to_si_renames_and_converts():
    out = loading.convert_to_si(make_df())
    assert out["OA_TEMP_C"].tolist() == pytest.approx([0.0, 100.0])
    assert out["OA_Ls"].iloc[0] == pytest.approx(47.195, abs=1e-3)
    assert "SA_SP_Pa" in out.columns


def test_convert_skips_missing_columns():
    out = loading.convert_to_si(pd.DataFrame({"OA_TEMP": [32.0]}))
    assert list(out.columns) == ["OA_TEMP_C"]


def test_audit_flags_constant_and_out_of_range():
    report = loading.audit_ahu(loading.convert_to_si(make_df()))
    assert "SF_SPD" in report["constant_columns"]
    assert report["out_of_range_0_1"] == {"CHWC_VLV": 1}
