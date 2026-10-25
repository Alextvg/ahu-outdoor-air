"""Estimate damper position from the mixing equation and test it on stuck files.

Run from the project root (run ``python -m scripts.run_damper_audit`` first and
check its verdict):

    python -m scripts.run_damper_model
    python -m scripts.run_damper_model --min-delta 2 --degree 1

Steps:
  1. Load the fault-free file, keep fan-on rows after the start-up settling
     period, and compute the outdoor air fraction from the three temperatures.
  2. Split the fault-free rows into train / validation / test blocks in time.
  3. Fit two estimators of damper position from that fraction on the training
     rows (see ``ahu_oa.damper_model``): the inverted linearised regression and
     a direct polynomial.
  4. Score both on the fault-free validation and test blocks against the true
     damper position, and on each stuck file against its known position.

Rows where the temperature gap is below ``--min-delta`` get no estimate, so the
tables report ``coverage`` (share scored) alongside the error, as in the
baseline.

Outputs (all under ``results/``):
  tables/damper_model_coefficients.csv, damper_model_stability_by_month.csv,
  damper_model_estimators.csv, damper_model_metrics.csv,
  damper_model_stuck_summary.csv, damper_model_stuck_by_month.csv
  figures/damper_model_fault_free.png, damper_model_stuck_<case>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import config
from ahu_oa import (damper_model, diagnostics, filtering, loading, metrics,
                    physics, plotting, settling, splits)


def prepare(path: Path, min_delta: float, settle: str):
    """Load one file and keep fan-on, settled, complete rows with ``f_phys``.

    Returns:
        Tuple ``(df, report)``: the filtered dataframe (with ``well_conditioned``
        and ``f_phys`` columns) and the filter report.
    """
    df = loading.load_ahu_csv(str(path), time_col=config.TIME_COL)
    df = df[~df.index.duplicated()].sort_index()
    ok = settling.settled_mask(df.index, df["SF_SPD_DM"] == 1, settle)
    df, report = filtering.apply_filters(
        df[ok], occupied_rule=config.DAMPER_OCCUPIED_RULE, min_delta=min_delta)
    df["f_phys"] = physics.oa_fraction_from_temps(
        df["MA_TEMP_C"], df["RA_TEMP_C"], df["OA_TEMP_C"], min_delta=min_delta)
    return df, report


def score_estimator(case: str, split: str, est, truth, df: pd.DataFrame) -> list:
    """Score one estimator on one block, by conditioning group."""
    d_hat = est.predict(df["f_phys"].to_numpy(float))
    wc = df["well_conditioned"].to_numpy(bool)
    return [{"case": case, "split": split, "estimator": est.name, "group": g, **s}
            for g, s in metrics.summarise_by_conditioning(truth, d_hat, wc).items()]


def stuck_detail(case: str, est, df: pd.DataFrame, position: float, tol: float):
    """Summary and by-month rows for one estimator on one stuck file."""
    wc = df[df["well_conditioned"]].copy()
    wc["d_hat"] = est.predict(wc["f_phys"].to_numpy(float))
    wc = wc.dropna(subset=["d_hat"])
    wc["err"] = wc["d_hat"] - position
    if wc.empty:
        return None, []
    q1, med, q3 = np.percentile(wc["d_hat"], [25, 50, 75])
    summary = {"case": case, "estimator": est.name, "position_known": position,
               "n_scored": len(wc), "implied_median": float(med),
               "implied_q1": float(q1), "implied_q3": float(q3),
               "mae": float(wc["err"].abs().mean()), "bias": float(wc["err"].mean()),
               "share_within_tol": float((wc["err"].abs() <= tol).mean())}
    by_month = [{"case": case, "estimator": est.name, "month": int(m), "n_scored": len(g),
                 "implied_median": float(g["d_hat"].median()),
                 "mae": float(g["err"].abs().mean()), "bias": float(g["err"].mean())}
                for m, g in wc.groupby(wc.index.month)]
    return summary, by_month


def make_fault_free_figure(est, test: pd.DataFrame, damper_col: str) -> Figure:
    """Estimated vs true damper position on the fault-free test block."""
    wc = test[test["well_conditioned"]]
    fig = Figure(figsize=(6, 4.8))
    ax = fig.subplots()
    plotting.plot_est_vs_true(
        ax, wc[damper_col], est.predict(wc["f_phys"].to_numpy(float)),
        xlabel=f"True damper position ({damper_col})", ylabel="Estimated damper position",
        title=f"Fault-free test block, {est.name}")
    fig.tight_layout()
    return fig


def make_stuck_figure(case: str, est, df: pd.DataFrame, position: float, min_delta: float) -> Figure:
    """Implied-position histogram and error vs temperature gap for one stuck file."""
    d_hat = est.predict(df["f_phys"].to_numpy(float))
    fig = Figure(figsize=(12, 4.8))
    ax1, ax2 = fig.subplots(1, 2)
    wc = df["well_conditioned"].to_numpy(bool)
    vals = d_hat[wc & np.isfinite(d_hat)]
    if vals.size:
        ax1.hist(vals, bins=60, color="tab:blue")
    ax1.axvline(position, color="r", linestyle="--", label=f"known position {position:g}")
    ax1.set_xlabel("Implied damper position")
    ax1.set_ylabel("Rows")
    ax1.set_title(f"{case}: well-conditioned rows, {est.name}")
    ax1.legend()
    plotting.plot_error_vs_delta(
        ax2, (df["OA_TEMP_C"] - df["RA_TEMP_C"]).abs(), d_hat - position, min_delta=min_delta,
        ylabel="Error, implied - known position", title=f"{case}: all usable rows")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    parser.add_argument("--settle", default=config.SETTLE)
    parser.add_argument("--degree", type=int, default=config.DIRECT_POLY_DEGREE)
    parser.add_argument("--tol", type=float, default=config.POSITION_TOL)
    args = parser.parse_args(argv)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    col = config.DAMPER_COL

    ff_path = args.data_dir / config.FILES["fault_free"]
    if not ff_path.exists():
        print(f"[problem] fault-free file {ff_path} not found.")
        return 1
    try:
        ff, report = prepare(ff_path, args.min_delta, args.settle)
    except KeyError as err:
        print(f"[problem] fault_free: column {err} not found. "
              "Run `python -m scripts.run_damper_audit` to see the real column names.")
        return 1
    print(f"[ok] fault_free: {report['n_input']} rows, {report['n_complete']} fan-on settled, "
          f"{100 * report['coverage_well_conditioned']:.1f}% well-conditioned")

    parts = splits.split_data(ff, method=config.SPLIT_METHOD)
    train = parts["train"]
    estimators = [damper_model.fit_inverted(train, col),
                  damper_model.fit_direct(train, col, args.degree)]
    pd.DataFrame([{"estimator": e.name, "n_train": e.n_train, "damper_col": col,
                   **{k: str(v) for k, v in e.params.items()}} for e in estimators]
                 ).to_csv(config.TABLES_DIR / "damper_model_estimators.csv", index=False)
    diagnostics.compare_all_vs_flagged(train, damper_col=col).to_csv(
        config.TABLES_DIR / "damper_model_coefficients.csv", index=False)
    diagnostics.coefficient_stability(ff, by="month", damper_col=col).to_csv(
        config.TABLES_DIR / "damper_model_stability_by_month.csv", index=False)

    metric_rows = []
    for split in ("val", "test"):
        block = parts[split]
        for e in estimators:
            metric_rows += score_estimator("fault_free", split, e, block[col].to_numpy(float), block)
    make_fault_free_figure(estimators[0], parts["test"], col).savefig(
        config.FIGURES_DIR / "damper_model_fault_free.png", dpi=150)

    summary_rows, month_rows = [], []
    for case, fname in config.STUCK_FILES.items():
        path = args.data_dir / fname
        if not path.exists():
            print(f"[skip] {case}: {path} not found")
            continue
        position = config.STUCK_POSITION.get(case)
        if position is None:
            print(f"[skip] {case}: no position in config.STUCK_POSITION")
            continue
        try:
            df, rep = prepare(path, args.min_delta, args.settle)
        except KeyError as err:
            print(f"[problem] {case}: column {err} not found. "
                  "Run `python -m scripts.run_damper_audit` to see the real column names.")
            return 1
        truth = np.full(len(df), float(position))
        for e in estimators:
            metric_rows += score_estimator(case, "stuck", e, truth, df)
            s, m = stuck_detail(case, e, df, position, args.tol)
            if s:
                summary_rows.append(s)
                month_rows += m
        make_stuck_figure(case, estimators[0], df, position, args.min_delta).savefig(
            config.FIGURES_DIR / f"damper_model_stuck_{case}.png", dpi=150)
        print(f"[ok] {case}: {rep['n_complete']} fan-on settled rows")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(config.TABLES_DIR / "damper_model_metrics.csv", index=False)
    summary_df, month_df = pd.DataFrame(summary_rows), pd.DataFrame(month_rows)
    summary_df.to_csv(config.TABLES_DIR / "damper_model_stuck_summary.csv", index=False)
    month_df.to_csv(config.TABLES_DIR / "damper_model_stuck_by_month.csv", index=False)

    fmt = lambda v: f"{v:.4g}"
    print("\nEstimators (trained on fault-free train block):")
    for e in estimators:
        print(f"  {e.name}: n_train={e.n_train}, {e.params}")
    print("\nDamper position error (fraction of full stroke):")
    print(metrics_df.drop(columns="n_total").to_string(index=False, float_format=fmt))
    if not summary_df.empty:
        print(f"\nStuck files, well-conditioned rows (tolerance +-{args.tol}):")
        print(summary_df.to_string(index=False, float_format=fmt))
    print(f"\nTables written to {config.TABLES_DIR}\nFigures written to {config.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
