"""Audit the raw LBNL AHU CSV files (report section 2.6).

Run from the project root:

    python -m scripts.run_audit
    python -m scripts.run_audit --data-dir path/to/csvs

What it does, for each file listed in ``config.FILES`` that exists:
  1. Checks the CSV header for the timestamp column and the signals the
     analysis needs. If anything is missing it prints the real column names so
     you can fix ``config.TIME_COL`` or the column lists, then skips that file.
  2. Runs ``loading.audit_ahu`` (missing values, constant columns, 0-1 range).
  3. Applies the row filters and records how many rows survive each stage.
  4. Checks the timestamps (range, step, duplicates, ordering).
  5. Compares the true flows (OA_Ls, SA_Ls) in each bias file with the
     fault-free file, to answer "are the ground-truth flows unchanged?".
  6. Fits supply flow against the fan command on fault-free occupied rows.

Outputs (all under ``results/``):
  tables/audit_summary.csv, tables/audit_details.json,
  tables/ground_truth_vs_fault_free.csv, tables/sf_cs_vs_supply_flow.csv,
  figures/sf_cs_vs_supply_flow.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import config
from ahu_oa import filtering, loading, plotting

# Signals the analysis needs, by their raw dataset names.
REQUIRED_RAW = ["SYS_CTL", "OA_TEMP", "RA_TEMP", "MA_TEMP", "SF_CS",
                "OA_DMPR_DM", "OA_CFM", "SA_CFM"]


def check_header(path: Path, occupied_rule: str):
    """Read only the CSV header and report which expected columns are absent.

    Args:
        path: CSV file.
        occupied_rule: Occupancy rule from the config; ``"fan_status"`` also
            needs ``SF_SPD_DM``.

    Returns:
        Tuple ``(columns, missing)`` of the real column names and the expected
        names that are not among them.
    """
    cols = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    expected = [config.TIME_COL] + REQUIRED_RAW
    if occupied_rule == "fan_status":
        expected.append("SF_SPD_DM")
    return cols, [c for c in expected if c not in cols]


def timestamp_info(index: pd.Index) -> dict:
    """Describe the time index: range, typical step, duplicates, ordering."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return {"time_start": None, "time_end": None, "median_step": None,
                "n_duplicate_timestamps": None, "is_sorted": None}
    step = pd.Series(index[1:] - index[:-1]).median()
    return {
        "time_start": str(index.min()),
        "time_end": str(index.max()),
        "median_step": str(step),
        "n_duplicate_timestamps": int(index.duplicated().sum()),
        "is_sorted": bool(index.is_monotonic_increasing),
    }


def compare_flows(reference: pd.DataFrame, other: pd.DataFrame, case: str) -> dict:
    """Compare true flows in a bias file against the fault-free file.

    Compared on the timestamps the two files share. Bias files come from
    separate simulation runs, so flows can differ if the controller reacts to
    the biased outdoor air temperature.
    """
    common = reference.index.intersection(other.index)
    row = {"case": case, "n_common_rows": int(len(common))}
    for col in ("OA_Ls", "SA_Ls"):
        if len(common) == 0:
            row[f"{col}_max_abs_diff"] = np.nan
            row[f"{col}_share_rows_differ"] = np.nan
            continue
        diff = (other.loc[common, col] - reference.loc[common, col]).abs()
        row[f"{col}_max_abs_diff"] = float(diff.max())
        row[f"{col}_share_rows_differ"] = float((diff > 1e-6).mean())
    return row


def fit_supply_flow(df_occupied: pd.DataFrame) -> dict:
    """Straight-line fit of supply flow (L/s) against the fan command SF_CS."""
    x = df_occupied["SF_CS"].to_numpy(float)
    y = df_occupied["SA_Ls"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return {"n": int(x.size), "slope_Ls_per_unit": np.nan,
                "intercept_Ls": np.nan, "r_squared": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return {"n": int(x.size), "slope_Ls_per_unit": float(slope),
            "intercept_Ls": float(intercept), "r_squared": r2}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    args = parser.parse_args(argv)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows, details = [], {}
    flows_by_case = {}          # only true flows are kept, to limit memory
    fault_free_occupied = None
    problems = 0

    for case, fname in config.FILES.items():
        path = args.data_dir / fname
        if not path.exists():
            print(f"[skip] {case}: {path} not found")
            continue

        cols, missing = check_header(path, config.OCCUPIED_RULE)
        if missing:
            problems += 1
            print(f"[problem] {case}: expected columns not found: {missing}")
            print(f"          columns in {fname}:\n          {cols}")
            print("          Fix config.TIME_COL or the column lists, then re-run.")
            continue

        df = loading.load_ahu_csv(str(path), time_col=config.TIME_COL)
        audit = loading.audit_ahu(df)
        filtered, report = filtering.apply_filters(
            df, occupied_rule=config.OCCUPIED_RULE, min_delta=args.min_delta)
        stamps = timestamp_info(df.index)

        details[case] = {"file": fname, "audit": audit, "filter_report": report,
                         "timestamps": stamps, "columns": list(df.columns)}
        summary_rows.append({
            "case": case,
            "n_rows": audit["n_rows"],
            "n_occupied": report["n_occupied"],
            "n_complete": report["n_complete"],
            "n_well_conditioned": report["n_well_conditioned"],
            "coverage_well_conditioned": report["coverage_well_conditioned"],
            "n_columns_with_missing": len(audit["missing_values"]),
            "constant_columns": ", ".join(audit["constant_columns"]),
            "out_of_range_0_1": json.dumps(audit["out_of_range_0_1"]),
            "median_step": stamps["median_step"],
            "n_duplicate_timestamps": stamps["n_duplicate_timestamps"],
        })
        flows_by_case[case] = df[["OA_Ls", "SA_Ls"]].copy()
        if case == "fault_free":
            fault_free_occupied = filtered
        print(f"[ok] {case}: {audit['n_rows']} rows, {report['n_occupied']} occupied, "
              f"{report['n_complete']} complete, "
              f"{100 * report['coverage_well_conditioned']:.1f}% well-conditioned")

    if not summary_rows:
        print("\nNo files were audited. Put the CSVs in", args.data_dir,
              "or pass --data-dir.")
        return 1

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.TABLES_DIR / "audit_summary.csv", index=False)
    (config.TABLES_DIR / "audit_details.json").write_text(
        json.dumps(details, indent=2, default=str))

    # Ground truth flows in the bias files vs the fault-free file.
    if "fault_free" in flows_by_case and len(flows_by_case) > 1:
        ref = flows_by_case["fault_free"]
        comp = pd.DataFrame([compare_flows(ref, f, c)
                             for c, f in flows_by_case.items() if c != "fault_free"])
        comp.to_csv(config.TABLES_DIR / "ground_truth_vs_fault_free.csv", index=False)
        print("\nTrue flows in bias files vs fault-free (0 = unchanged):")
        print(comp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        print("\n[note] Need the fault-free file plus a bias file to compare true flows.")

    # Supply flow vs fan command (fault-free, occupied rows).
    if fault_free_occupied is not None and len(fault_free_occupied):
        fit = fit_supply_flow(fault_free_occupied)
        pd.DataFrame([fit]).to_csv(config.TABLES_DIR / "sf_cs_vs_supply_flow.csv", index=False)
        fig = Figure(figsize=(6, 4.5))
        ax = fig.subplots()
        plotting.plot_est_vs_true(
            ax, fault_free_occupied["SF_CS"], fault_free_occupied["SA_Ls"],
            xlabel="Supply fan command SF_CS", ylabel="Supply airflow SA (L/s)",
            title=f"Fault-free, occupied (linear R² = {fit['r_squared']:.3f})",
            one_to_one=False)
        xs = np.array([fault_free_occupied["SF_CS"].min(), fault_free_occupied["SF_CS"].max()])
        ax.plot(xs, fit["slope_Ls_per_unit"] * xs + fit["intercept_Ls"], "k--",
                linewidth=1, label="linear fit")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / "sf_cs_vs_supply_flow.png", dpi=150)
        print(f"\nSupply flow vs SF_CS (fault-free, occupied): R² = {fit['r_squared']:.4f}, "
              f"slope = {fit['slope_Ls_per_unit']:.4g} L/s per unit command")

    print("\nAudit summary:")
    print(summary.drop(columns=["constant_columns", "out_of_range_0_1"])
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nTables written to {config.TABLES_DIR}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
