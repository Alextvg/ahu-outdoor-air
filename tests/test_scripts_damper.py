"""Smoke tests: run the two damper scripts end to end on synthetic CSVs.

Needs the project root on the path (run ``python -m pytest`` from the root);
skipped otherwise, because the scripts import ``config``.
"""
import shutil

import pandas as pd
import pytest

config = pytest.importorskip("config")
from scripts import run_damper_audit, run_damper_model  # noqa: E402

from synth_data import build_csv  # noqa: E402


@pytest.fixture
def project(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    build_csv(data / "AHU_annual.csv")
    build_csv(data / "damper_stuck_100_annual_short.csv", stuck=1.0,
              start="2020-04-01", end="2020-11-30 23:00", seed=1)
    shutil.copy(data / "AHU_annual.csv", data / "oa_bias_2_annual.csv")
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path / "results" / "tables")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "results" / "figures")
    monkeypatch.setattr(config, "FILES", {"fault_free": "AHU_annual.csv",
                                          "oa_bias_+2": "oa_bias_2_annual.csv"})
    return data


def test_audit_script_writes_one_report_and_figures(project):
    assert run_damper_audit.main(["--data-dir", str(project)]) == 0
    report = config.TABLES_DIR / "damper_audit_report.txt"
    assert report.exists() and "VERDICT: FIT" in report.read_text()
    for fig in ("damper_audit_fraction_vs_damper.png", "damper_audit_monthly_rows.png"):
        assert (config.FIGURES_DIR / fig).stat().st_size > 0
    assert not (config.TABLES_DIR / "damper_audit_checks.csv").exists()   # tables are opt-in


def test_audit_script_tables_flag_writes_csvs(project):
    run_damper_audit.main(["--data-dir", str(project), "--tables"])
    for name in ("files", "structure", "monthly", "fits", "stuck", "checks"):
        assert (config.TABLES_DIR / f"damper_audit_{name}.csv").exists()
    checks = pd.read_csv(config.TABLES_DIR / "damper_audit_checks.csv")
    assert "FAIL" not in set(checks["status"])


def test_audit_script_returns_1_when_a_check_fails(project):
    shutil.copy(project / "AHU_annual.csv", project / "damper_stuck_100_annual_short.csv")
    assert run_damper_audit.main(["--data-dir", str(project)]) == 1


def test_audit_script_reports_missing_columns_and_skips(project, capsys):
    pd.read_csv(project / "AHU_annual.csv").drop(columns="OA_DMPR").to_csv(
        project / "AHU_annual.csv", index=False)
    assert run_damper_audit.main(["--data-dir", str(project)]) == 1
    out = capsys.readouterr().out
    assert "OA_DMPR" in out and "columns in AHU_annual.csv" in out


def test_audit_script_missing_fault_free_file_returns_1(project):
    (project / "AHU_annual.csv").unlink()
    assert run_damper_audit.main(["--data-dir", str(project)]) == 1


def test_model_script_writes_tables_and_recovers_known_position(project):
    assert run_damper_model.main(["--data-dir", str(project)]) == 0
    for name in ("estimators", "metrics", "stuck_summary", "stuck_by_month",
                 "coefficients", "stability_by_month"):
        assert (config.TABLES_DIR / f"damper_model_{name}.csv").exists()
    assert (config.FIGURES_DIR / "damper_model_fault_free.png").stat().st_size > 0
    assert (config.FIGURES_DIR / "damper_model_stuck_damper_stuck_100.png").stat().st_size > 0
    s = pd.read_csv(config.TABLES_DIR / "damper_model_stuck_summary.csv")
    assert (s["implied_median"] - s["position_known"]).abs().max() < 0.05


def test_model_script_reports_coverage_for_ill_conditioned_rows(project):
    run_damper_model.main(["--data-dir", str(project)])
    m = pd.read_csv(config.TABLES_DIR / "damper_model_metrics.csv")
    ill = m[(m["group"] == "ill_conditioned") & (m["split"] == "stuck")]
    assert (ill["coverage"] == 0).all()          # no estimate where delta is small


def test_model_script_skips_missing_stuck_file(project, capsys):
    (project / "damper_stuck_100_annual_short.csv").unlink()
    assert run_damper_model.main(["--data-dir", str(project)]) == 0
    assert "[skip] damper_stuck_100" in capsys.readouterr().out
