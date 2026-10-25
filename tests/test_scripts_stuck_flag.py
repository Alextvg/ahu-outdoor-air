"""Smoke tests: run scripts/run_stuck_flag.py end to end on synthetic CSVs."""
import pandas as pd
import pytest

config = pytest.importorskip("config")
from scripts import run_stuck_flag  # noqa: E402

from synth_data import build_csv  # noqa: E402

ARGS = ["--knots", "10", "--window", "12h", "--min-rows", "3"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    build_csv(data / "AHU_annual.csv")
    build_csv(data / "damper_stuck_100_annual_short.csv", stuck=1.0,
              start="2020-04-01", end="2020-11-30 23:00", seed=1)
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path / "results" / "tables")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "results" / "figures")
    monkeypatch.setattr(config, "FILES", {"fault_free": "AHU_annual.csv"})
    return data


def run(project, *extra):
    assert run_stuck_flag.main(["--data-dir", str(project), *ARGS, *extra]) == 0
    t = config.TABLES_DIR
    return pd.read_csv(t / "stuck_flag_rates.csv"), pd.read_csv(t / "stuck_flag_sweep.csv")


def test_few_false_alarms_and_the_stuck_file_is_caught(project):
    rates, _ = run(project)
    fa = rates[rates["case"] == "fault_free_test"].iloc[0]
    det = rates[rates["case"] == "damper_stuck_100"].iloc[0]
    assert fa["role"] == "false_alarm" and fa["row_flag_rate"] < 0.05
    assert det["role"] == "detection" and det["row_flag_rate"] > 0.8
    assert det["miss_row_rate"] == pytest.approx(1 - det["row_flag_rate"])
    assert det["row_flag_rate"] <= det["detectable_share"] + 0.05      # roughly capped by what is detectable


def test_threshold_is_chosen_on_validation_only(project):
    run(project)
    s = pd.read_csv(config.TABLES_DIR / "stuck_flag_settings.csv").set_index("setting")["value"]
    assert "validation" in s["threshold_chosen_by"] and s["estimator"].startswith("monotone_spline_uniform_10")


def test_fixed_threshold_overrides_the_choice(project):
    rates, sweep = run(project, "--threshold", "0.12")
    assert (rates["threshold"] == 0.12).all()
    assert 0.12 in set(sweep["threshold"])


def test_sweep_flag_rate_never_rises_with_threshold(project):
    _, sweep = run(project)
    for _, g in sweep.groupby("case"):
        r = g.sort_values("threshold")["row_flag_rate"].dropna().to_numpy()
        assert (r[1:] <= r[:-1] + 1e-12).all()


def test_cubic_variant_and_outputs(project):
    rates, _ = run(project, "--variant", "cubic", "--rule", "quantile")
    assert set(rates["role"]) == {"false_alarm", "detection"}
    for name in ("stuck_flag_by_month.csv", "stuck_flag_settings.csv"):
        assert (config.TABLES_DIR / name).exists()
    assert (config.FIGURES_DIR / "stuck_flag_tradeoff.png").stat().st_size > 0


def test_missing_fault_free_file_returns_error(project):
    (project / "AHU_annual.csv").unlink()
    assert run_stuck_flag.main(["--data-dir", str(project), *ARGS]) == 1
