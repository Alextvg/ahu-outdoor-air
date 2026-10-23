"""Check whether the files can answer the damper-stuck question.

Run from the project root:

    python -m scripts.run_damper_audit
    python -m scripts.run_damper_audit --data-dir path/to/csvs --settle 15min

The design: learn damper position from the mixing equation on the fault-free
file, then test it on files where the damper is stuck at a known position.
This script decides whether the data can support that, before any model is
fitted. For the fault-free file and each file in ``config.STUCK_FILES`` it:
  1. Checks the CSV header for the time column and the signals the design
     needs (the outdoor flow column OA_CFM is not needed). If anything is
     missing it prints the real column names and skips that file.
  2. Checks that the stuck files are not copies of the fault-free file, that
     sampling is regular, and how many usable rows there are per month.
  3. Fits the outdoor air fraction against the damper on fault-free data and
     asks how well damper position can be recovered.
  4. For each stuck file, checks that a damper column really is constant, that
     it matches the documented position, that the fault-free data covers that
     position, and that the mixing equation implies that position.
  5. Writes a PASS / WARN / FAIL checklist and a one-line verdict.

The bias files in ``config.FILES`` are included only to detect duplicates.

Outputs (all under ``results/``):
  tables/damper_audit_report.txt   <- the one short file to share
  figures/damper_audit_fraction_vs_damper.png, damper_audit_monthly_rows.png
  tables/damper_audit_<name>.csv (files, structure, monthly, fits, stuck,
      checks) only when run with --tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import config
from ahu_oa import audit as audit_mod
from ahu_oa import plotting

# Signals the design needs, by their raw dataset names (no flow columns).
REQUIRED_RAW = ["OA_TEMP", "RA_TEMP", "MA_TEMP", "SF_SPD_DM", "OA_DMPR", "OA_DMPR_DM"]


def check_header(path: Path):
    """Read only the CSV header and report which expected columns are absent.

    Returns:
        Tuple ``(columns, missing)`` of the real column names and the expected
        names that are not among them.
    """
    cols = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    expected = [config.TIME_COL] + REQUIRED_RAW
    return cols, [c for c in expected if c not in cols]


def make_fraction_figure(result, damper_col: str) -> Figure:
    """Fault-free fraction vs damper with the fit line and each stuck file's median."""
    wc = result.prepared["fault_free"]
    wc = wc[wc["well_conditioned"]]
    fit = result.fits[result.fits["damper_col"] == damper_col].iloc[0]
    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()
    plotting.plot_est_vs_true(
        ax, wc[damper_col], wc["f_phys"], xlabel=f"Damper position ({damper_col})",
        ylabel="Outdoor air fraction from mixing equation",
        title=f"Fault-free, well-conditioned (R² = {fit['r_squared']:.3f})", one_to_one=False)
    xs = np.array([0.0, 1.0])
    ax.plot(xs, fit["a"] + fit["b"] * xs, "k--", linewidth=1,
            label=f"fit: {fit['a']:.3f} + {fit['b']:.3f}·damper")
    for _, r in result.stuck.iterrows():
        if pd.notna(r.get("fraction_mean")):
            ax.errorbar([r["position_known"]], [r["fraction_mean"]], yerr=[r["fraction_sd"]],
                        fmt="o", color="red", capsize=3, label=f"{r['case']} (mean ± sd)")
    # plot_est_vs_true limits the view to the bulk of the data; leave room so
    # markers on the edge (a stuck position of 1.0) are not clipped.
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    ax.set_xlim(min(x_lo, -0.03), max(x_hi, 1.03))
    ax.set_ylim(y_lo, y_hi + 0.05 * (y_hi - y_lo))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def make_monthly_figure(result) -> Figure:
    """Well-conditioned usable rows per month for each analysed case."""
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    cases = list(result.structure["case"])
    width = 0.8 / max(len(cases), 1)
    for i, case in enumerate(cases):
        m = result.monthly[result.monthly["case"] == case].set_index("month")["n_well_conditioned"]
        m = m.reindex(range(1, 13), fill_value=0)
        ax.bar(np.arange(1, 13) + i * width, m.to_numpy(), width=width, label=case)
    ax.set_xticks(np.arange(1, 13) + 0.4 - width / 2)
    ax.set_xticklabels(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Well-conditioned usable rows")
    ax.set_title("Data available for the mixing equation, by month")
    ax.legend()
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    parser.add_argument("--settle", default=config.SETTLE)
    parser.add_argument("--tables", action="store_true",
                        help="also write the full CSV tables")
    args = parser.parse_args(argv)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ff_path = args.data_dir / config.FILES["fault_free"]
    if not ff_path.exists():
        print(f"[problem] fault-free file {ff_path} not found. "
              "Put the CSVs in", args.data_dir, "or pass --data-dir.")
        return 1

    stuck, extra, problems = {}, {}, 0
    for case, fname in [("fault_free", config.FILES["fault_free"]), *config.STUCK_FILES.items()]:
        path = args.data_dir / fname
        if not path.exists():
            print(f"[skip] {case}: {path} not found")
            continue
        cols, missing = check_header(path)
        if missing:
            problems += 1
            print(f"[problem] {case}: expected columns not found: {missing}")
            print(f"          columns in {fname}:\n          {cols}")
            print("          Fix config.TIME_COL or REQUIRED_RAW, then re-run.")
            continue
        if case != "fault_free":
            stuck[case] = (path, config.STUCK_POSITION.get(case))
    if problems:
        return 1
    if not stuck:
        print("\nNo stuck files found in", args.data_dir, "- nothing to audit.")
        return 1
    for case, fname in config.FILES.items():           # duplicate check only
        path = args.data_dir / fname
        if case != "fault_free" and path.exists():
            extra[case] = path

    result = audit_mod.audit(ff_path, stuck, extra, time_col=config.TIME_COL,
                             min_delta=args.min_delta, settle=args.settle)

    report = audit_mod.format_report(result)
    (config.TABLES_DIR / "damper_audit_report.txt").write_text(report)
    if args.tables:
        for name in ("files", "structure", "monthly", "fits", "stuck", "checks"):
            getattr(result, name).to_csv(
                config.TABLES_DIR / f"damper_audit_{name}.csv", index=False)

    if not result.fits.empty:
        col = (config.DAMPER_COL if config.DAMPER_COL in set(result.fits["damper_col"])
               else result.fits.loc[result.fits["r_squared"].idxmax(), "damper_col"])
        make_fraction_figure(result, col).savefig(
            config.FIGURES_DIR / "damper_audit_fraction_vs_damper.png", dpi=150)
    make_monthly_figure(result).savefig(
        config.FIGURES_DIR / "damper_audit_monthly_rows.png", dpi=150)

    print(report)
    print(f"Report written to {config.TABLES_DIR / 'damper_audit_report.txt'}")
    return 1 if (result.checks["status"] == "FAIL").any() else 0


if __name__ == "__main__":
    sys.exit(main())
