import shutil

import pytest

from ahu_oa.audit import audit, format_report
from synth_data import build_csv


def status(result, name_part):
    c = result.checks
    return list(c[c["check"].str.contains(name_part, regex=False)]["status"])


@pytest.fixture
def ff(tmp_path):
    return build_csv(tmp_path / "ff.csv")


def stuck_files(tmp_path, pos_doc=1.0, name="damper_stuck_100_annual_short.csv", **kw):
    p = build_csv(tmp_path / name, stuck=kw.pop("stuck", 1.0),
                  start="2020-04-01", end="2020-11-30 23:00", seed=1, **kw)
    return {"stuck": (p, pos_doc)}


def test_good_dataset_is_fit(tmp_path, ff):
    r = audit(ff, stuck_files(tmp_path))
    assert r.verdict.startswith("FIT")
    assert "FAIL" not in set(r.checks["status"])


def test_result_tables_are_dataframes_with_expected_columns(tmp_path, ff):
    r = audit(ff, stuck_files(tmp_path))
    assert {"case", "n_usable", "n_well_conditioned"} <= set(r.structure.columns)
    assert {"damper_col", "a", "b", "r_squared"} <= set(r.fits.columns)
    assert {"implied_median", "position_known", "stuck_col"} <= set(r.stuck.columns)
    assert set(r.monthly["case"]) == {"fault_free", "stuck"}
    assert set(r.prepared) == {"fault_free", "stuck"}


def test_stuck_file_identical_to_fault_free_fails(tmp_path, ff):
    dup = tmp_path / "damper_stuck_100_copy.csv"
    shutil.copy(ff, dup)
    r = audit(ff, {"stuck": (str(dup), 1.0)})
    assert "FAIL" in status(r, "differ from fault-free")
    assert r.verdict.startswith("NOT FIT")


def test_duplicate_extra_files_are_reported(tmp_path, ff):
    a, b = tmp_path / "oa_bias_a.csv", tmp_path / "oa_bias_b.csv"
    shutil.copy(ff, a)
    shutil.copy(ff, b)
    r = audit(ff, stuck_files(tmp_path), {"bias_a": str(a), "bias_b": str(b)})
    assert "INFO" in status(r, "duplicate files")
    row = r.files[r.files["case"] == "bias_a"].iloc[0]
    assert "bias_b" in row["same_bytes_as"]
    assert bool(row["same_numeric_data_as_fault_free"])


def test_non_constant_damper_means_stuck_state_not_recorded(tmp_path, ff):
    r = audit(ff, stuck_files(tmp_path, constant_col=False))
    assert "FAIL" in status(r, "stuck state visible")


def test_wrong_physics_for_documented_position_fails(tmp_path, ff):
    # Recorded as stuck at 1.0, but the mixed air behaves like position 0.2.
    r = audit(ff, stuck_files(tmp_path, stuck=0.2, label_pos=1.0))
    assert "FAIL" in status(r, "implied position matches known")
    assert r.verdict.startswith("NOT FIT")


def test_documented_position_mismatch_warns(tmp_path, ff):
    r = audit(ff, stuck_files(tmp_path, pos_doc=0.5))
    assert "WARN" in status(r, "documented position matches data")


def test_position_is_read_from_filename_when_not_documented(tmp_path, ff):
    r = audit(ff, stuck_files(tmp_path, pos_doc=None))
    row = r.stuck.iloc[0]
    assert row["position_source"] == "filename"
    assert row["position_known"] == pytest.approx(1.0)


def test_position_not_covered_by_training_warns(tmp_path):
    ff_path = build_csv(tmp_path / "ff.csv", cap_damper=0.6)
    r = audit(ff_path, stuck_files(tmp_path))
    assert "WARN" in status(r, "training covers stuck position")


def test_constant_oa_flow_is_noted(tmp_path, ff):
    r = audit(ff, stuck_files(tmp_path))
    assert "INFO" in status(r, "outdoor flow constant")


def test_weather_match_is_reported(tmp_path, ff):
    r = audit(ff, stuck_files(tmp_path))
    detail = r.checks[r.checks["check"].str.contains("weather matches")]["detail"].iloc[0]
    assert "identical" in detail


def test_report_is_short_and_leads_with_the_verdict(tmp_path, ff):
    text = format_report(audit(ff, stuck_files(tmp_path)))
    lines = text.splitlines()
    assert lines[1].startswith("VERDICT: FIT")
    assert len(lines) < 60
    for part in ("FILES", "DATA", "FAULT-FREE", "STUCK: stuck", "CHECKS"):
        assert part in text


def test_report_lists_failures_and_collapses_duplicates(tmp_path, ff):
    dup = tmp_path / "damper_stuck_100_copy.csv"
    shutil.copy(ff, dup)
    other = tmp_path / "oa_bias_a.csv"
    shutil.copy(ff, other)
    r = audit(ff, {"stuck": (str(dup), 1.0)}, {"bias_a": str(other)})
    text = format_report(r)
    assert "[FAIL] stuck files differ from fault-free" in text
    assert text.count("IDENTICAL FILES") == 1
    assert "fault_free = " in text and "bias_a" in text


def test_report_handles_a_stuck_file_with_no_constant_column(tmp_path, ff):
    text = format_report(audit(ff, stuck_files(tmp_path, constant_col=False)))
    assert "stuck state not recorded" in text


def test_report_is_plain_ascii_so_any_console_or_editor_can_show_it(tmp_path, ff):
    assert format_report(audit(ff, stuck_files(tmp_path))).isascii()
