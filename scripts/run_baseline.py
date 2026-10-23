"""Score the raw mixing-equation baseline against the simulator ground truth.

Run from the project root:

    python -m scripts.run_baseline
    python -m scripts.run_baseline --min-delta 2 --cases fault_free oa_bias_+2

The baseline needs no training, so it can be scored on all five files now:
the fault-free file and the four outdoor-air-sensor-bias files. This gives the
reference numbers that the regression model (and the bias experiment) must beat.

Important choice: the supply airflow used to turn the fraction into a flow is
the TRUE supply airflow (SA_Ls). That isolates the error of step 1 (the outdoor
air fraction). Once a supply flow model exists, its added error is measured
separately (report section 3.5).

How ill-conditioned rows are handled: the fraction is computed on every
occupied row (with only a tiny threshold to avoid dividing by zero), and rows
are then labelled well- or ill-conditioned using ``--min-delta``. The metrics
are reported for all rows and for each group, and the compliance check is made
only on well-conditioned rows, which are the rows the baseline would report.

Outputs (under ``results/``):
  tables/baseline_metrics.csv, tables/baseline_diagnostics.csv,
  tables/baseline_compliance.csv, figures/baseline_<case>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import config
from ahu_oa import filtering, loading, metrics, physics, plotting

# Only prevents division by exactly zero; not a conditioning filter.
RAW_MIN_DELTA_K = 1e-3


def score_case(case: str, path: Path, min_delta: float, required_ls: float):
    """Load one file, run the baseline and score it.

    Args:
        case: Case label (used in the tables).
        path: CSV file to load.
        min_delta: Ill-conditioning threshold (K).
        required_ls: Ventilation minimum Vbz (L/s) for the compliance check.

    Returns:
        Tuple ``(metric_rows, diagnostics_row, compliance_row, data)`` where
        ``data`` is a dataframe of the arrays needed for plotting.
    """
    df = loading.load_ahu_csv(str(path), time_col=config.TIME_COL)
    df, report = filtering.apply_filters(
        df, occupied_rule=config.OCCUPIED_RULE, min_delta=min_delta)

    # Step 1: outdoor air fraction from the three temperatures (baseline).
    f_hat = physics.oa_fraction_from_temps(
        df["MA_TEMP_C"], df["RA_TEMP_C"], df["OA_TEMP_C"], min_delta=RAW_MIN_DELTA_K)
    # Step 2 is skipped on purpose: use the true supply flow (see module docstring).
    oa_hat = physics.oa_flow_from_fraction(f_hat, df["SA_Ls"])

    oa_true = df["OA_Ls"].to_numpy(float)
    sa_true = df["SA_Ls"].to_numpy(float)
    f_true = np.divide(oa_true, sa_true, out=np.full_like(oa_true, np.nan),
                       where=sa_true > 0)
    wc = df["well_conditioned"].to_numpy(bool)

    rows = []
    for quantity, truth, est in (("oa_flow_Ls", oa_true, oa_hat),
                                 ("oa_fraction", f_true, f_hat)):
        for group, summary in metrics.summarise_by_conditioning(truth, est, wc).items():
            rows.append({"case": case, "quantity": quantity, "group": group, **summary})

    plausible = physics.fraction_plausible(f_hat)
    diagnostics = {
        "case": case,
        "bias_K": config.BIAS_K.get(case),
        "n_complete": report["n_complete"],
        "coverage_well_conditioned": report["coverage_well_conditioned"],
        "share_fraction_plausible_all": float(plausible.mean()) if plausible.size else np.nan,
        "share_fraction_plausible_well": float(plausible[wc].mean()) if wc.any() else np.nan,
        "min_delta_K": min_delta,
    }

    # Compliance: only where the baseline would report a value (well-conditioned).
    compliance = {"case": case, "required_Ls": required_ls,
                  **metrics.compliance_rates(oa_true[wc], oa_hat[wc], required_ls)}

    data = pd.DataFrame({
        "oa_true": oa_true, "oa_hat": oa_hat,
        "delta_k": (df["OA_TEMP_C"] - df["RA_TEMP_C"]).abs().to_numpy(float),
        "well_conditioned": wc,
    })
    return rows, diagnostics, compliance, data


def make_figure(case: str, data: pd.DataFrame, min_delta: float) -> Figure:
    """Two-panel diagnostic figure: estimate vs truth, and error vs temperature gap."""
    wc = data["well_conditioned"].to_numpy(bool)
    fig = Figure(figsize=(12, 4.8))
    ax1, ax2 = fig.subplots(1, 2)
    plotting.plot_est_vs_true(
        ax1, data.loc[wc, "oa_true"], data.loc[wc, "oa_hat"],
        xlabel="True outdoor airflow (L/s)", ylabel="Estimated outdoor airflow (L/s)",
        title=f"{case}: well-conditioned rows")
    plotting.plot_error_vs_delta(
        ax2, data["delta_k"], data["oa_hat"] - data["oa_true"], min_delta=min_delta,
        ylabel="Error, estimate - true (L/s)", title=f"{case}: all occupied rows")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    parser.add_argument("--cases", nargs="+", default=list(config.FILES),
                        choices=list(config.FILES), help="subset of cases to run")
    args = parser.parse_args(argv)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    people = physics.default_occupancy(config.FLOOR_AREA_M2, config.PEOPLE_PER_1000FT2)
    required = physics.breathing_zone_flow_ls(config.FLOOR_AREA_M2, people)
    print(f"Ventilation minimum Vbz = {required:.0f} L/s "
          f"({people:.0f} people assumed, lower bound)")

    metric_rows, diag_rows, comp_rows = [], [], []
    for case in args.cases:
        path = args.data_dir / config.FILES[case]
        if not path.exists():
            print(f"[skip] {case}: {path} not found")
            continue
        try:
            rows, diag, comp, data = score_case(case, path, args.min_delta, required)
        except KeyError as err:
            print(f"[problem] {case}: column {err} not found. "
                  "Run `python -m scripts.run_audit` to see the real column names.")
            return 1
        metric_rows += rows
        diag_rows.append(diag)
        comp_rows.append(comp)
        make_figure(case, data, args.min_delta).savefig(
            config.FIGURES_DIR / f"baseline_{case}.png", dpi=150)
        print(f"[ok] {case}")

    if not metric_rows:
        print("\nNo files were scored. Put the CSVs in", args.data_dir, "or pass --data-dir.")
        return 1

    metrics_df = pd.DataFrame(metric_rows)
    diag_df = pd.DataFrame(diag_rows)
    comp_df = pd.DataFrame(comp_rows)
    metrics_df.to_csv(config.TABLES_DIR / "baseline_metrics.csv", index=False)
    diag_df.to_csv(config.TABLES_DIR / "baseline_diagnostics.csv", index=False)
    comp_df.to_csv(config.TABLES_DIR / "baseline_compliance.csv", index=False)

    fmt = lambda v: f"{v:.4g}"
    shown = metrics_df[metrics_df["quantity"] == "oa_flow_Ls"].drop(columns="quantity")
    print("\nOutdoor airflow error (L/s), baseline with true supply flow:")
    print(shown.to_string(index=False, float_format=fmt))
    print("\nDiagnostics:")
    print(diag_df.to_string(index=False, float_format=fmt))
    print("\nCompliance decision vs truth (well-conditioned rows):")
    print(comp_df.to_string(index=False, float_format=fmt))
    print(f"\nTables written to {config.TABLES_DIR}\nFigures written to {config.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
