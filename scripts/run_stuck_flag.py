"""Stuck-damper flag using a 45-knot spline: false alarms and detections.

Run from the project root (audit first, see ``scripts/run_damper_audit.py``):

    python -m scripts.run_stuck_flag
    python -m scripts.run_stuck_flag --variant cubic --rule quantile
    python -m scripts.run_stuck_flag --threshold 0.1 --window 120min

What it does
  1. Loads the fault-free file, splits it into time blocks (``ahu_oa.splits``) and fits one
     spline (default: monotone, uniform knots, 45 knots) on the TRAIN block.
  2. Estimates the damper position from the temperatures only and compares it with the
     commanded position ``OA_DMPR_DM`` (``ahu_oa.stuck_flag``). A row is flagged when the
     median gap over the trailing window exceeds a threshold.
  3. Chooses the threshold on the fault-free VALIDATION block: the smallest candidate whose
     false-alarm rate stays under ``--max-false-alarm``. Pass ``--threshold`` to fix it.
  4. Reports, at that threshold:
       false-alarm rate  = share of fault-free TEST rows (and days) that are flagged
                           (the validation rate is also shown, but it chose the threshold)
       detection rate    = share of each stuck file's rows (and days) that are flagged
       miss rate         = 1 - detection rate
     No stuck file is used to choose any setting.

Outputs (written under config.TABLES_DIR and config.FIGURES_DIR)
  tables/stuck_flag_rates.csv      false alarms (fault_free_val, fault_free_test) and detections
                                   (each stuck file) at the chosen threshold
  tables/stuck_flag_sweep.csv      the same for every candidate threshold (trade-off)
  tables/stuck_flag_by_month.csv   detection rate by month for each stuck file
  tables/stuck_flag_settings.csv   estimator, window, threshold and the command check
  figures/stuck_flag_tradeoff.png  flag rate against threshold, chosen threshold marked

Reading the result: ``detectable_share`` is the share of a stuck file's rows where the command
differs from the stuck position by more than the threshold. Detection is roughly capped by it (the
trailing window smears the edges slightly), so a low value means the controller mostly asks for the
stuck position anyway and the damper looks healthy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import config
from ahu_oa import spline_model, splits, stuck_flag
from scripts.run_damper_model import prepare

DEFAULT_THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def estimate_and_gap(est, df: pd.DataFrame) -> pd.Series:
    """Gap between the spline estimate and the command on well-conditioned rows.

    Args:
        est: Fitted spline (anything with ``predict``).
        df: Dataframe from ``prepare`` with ``f_phys``, ``well_conditioned`` and ``OA_DMPR_DM``.

    Returns:
        ``|estimate - command|`` indexed by time, rows without an estimate dropped.
    """
    wc = df[df["well_conditioned"]].dropna(subset=["f_phys", "OA_DMPR_DM"])
    pred = pd.Series(est.predict(wc["f_phys"].to_numpy(float)), index=wc.index)
    return stuck_flag.disagreement(pred, wc["OA_DMPR_DM"])


def rates_row(case: str, split: str, role: str, flags: pd.DataFrame, threshold: float,
              detectable_share: float = np.nan) -> dict:
    """One results row: rates for a case, with its role (false alarm or detection)."""
    r = stuck_flag.flag_rates(flags)
    row = {"case": case, "split": split, "role": role, "threshold": threshold, **r,
           "coverage": r["n_decided"] / r["n_rows"] if r["n_rows"] else np.nan}
    if role == "detection":
        row["miss_row_rate"] = 1.0 - r["row_flag_rate"] if np.isfinite(r["row_flag_rate"]) else np.nan
        row["miss_day_rate"] = 1.0 - r["day_flag_rate"] if np.isfinite(r["day_flag_rate"]) else np.nan
        row["detectable_share"] = detectable_share
    return row


def make_tradeoff_figure(sweep: pd.DataFrame, threshold: float, title: str) -> Figure:
    """Row flag rate against threshold: false alarms (fault-free test) and detections."""
    fig = Figure(figsize=(8, 4.8))
    ax = fig.subplots()
    for case, g in sweep.groupby("case"):
        g = g.sort_values("threshold")
        fa = case.startswith("fault_free")
        tuned = case == "fault_free_val"
        ax.plot(g["threshold"], g["row_flag_rate"], marker="o", linestyle=":" if tuned else ("--" if fa else "-"),
                color=("grey" if tuned else "black") if fa else None,
                label=("fault-free validation (threshold chosen here)" if tuned else
                       "fault-free test (false alarms)") if fa else f"{case} (detection)")
    ax.axvline(threshold, color="grey", linestyle=":", label=f"chosen threshold {threshold:g}")
    ax.set_xlabel("Threshold on median |estimate - command| (damper units)")
    ax.set_ylabel("Share of rows flagged")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    parser.add_argument("--settle", default=config.SETTLE)
    parser.add_argument("--variant", choices=("cubic", "monotone"), default="monotone",
                        help="spline variant used as the estimator")
    parser.add_argument("--rule", choices=list(spline_model.KNOT_RULES), default="uniform",
                        help="knot rule")
    parser.add_argument("--knots", type=int, default=45, help="number of knots")
    parser.add_argument("--window", default="60min",
                        help="trailing window for the median gap (pandas offset string)")
    parser.add_argument("--min-rows", type=int, default=20,
                        help="fewest rows in the window for a decision")
    parser.add_argument("--threshold", type=float, default=None,
                        help="fix the threshold instead of choosing it on validation")
    parser.add_argument("--max-false-alarm", type=float, default=0.01,
                        help="highest fault-free validation false-alarm rate allowed")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS),
                        help="candidate thresholds for the choice and the sweep")
    args = parser.parse_args(argv)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    col = config.DAMPER_COL
    cands = sorted(set(args.thresholds) | ({args.threshold} if args.threshold is not None else set()))

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
    print(f"[ok] fault_free: {report['n_complete']} fan-on settled rows")

    parts = splits.split_data(ff, method=config.SPLIT_METHOD)
    train, val, test = parts["train"], parts["val"], parts["test"]
    try:
        est = spline_model.fit_spline(train, col, n_knots=args.knots, rule=args.rule,
                                      monotone=args.variant == "monotone")
    except ValueError as err:
        print(f"[problem] could not fit {args.variant} spline with {args.knots} knots: {err}")
        return 1
    print(f"[ok] fitted {est.name} on {est.n_train} training rows")

    gap_val, gap_test = estimate_and_gap(est, val), estimate_and_gap(est, test)
    if args.threshold is None:
        threshold = stuck_flag.choose_threshold(gap_val, cands, args.window, args.min_rows,
                                                args.max_false_alarm)
        how = f"smallest candidate with validation false alarms <= {args.max_false_alarm:.1%}"
    else:
        threshold, how = args.threshold, "fixed with --threshold"
    print(f"[ok] threshold {threshold:g} ({how})")

    # ---- gaps for every case (fault-free test and each stuck file)
    gaps = {("fault_free_val", "val"): (gap_val, None), ("fault_free_test", "test"): (gap_test, None)}
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
            print(f"[problem] {case}: column {err} not found.")
            return 1
        gaps[(case, "stuck")] = (estimate_and_gap(est, df), (df, float(position)))
        print(f"[ok] {case}: {rep['n_complete']} fan-on settled rows")

    # ---- rates at the chosen threshold, and the sweep over all candidates
    rate_rows, sweep_rows, month_rows = [], [], []
    for (case, split), (gap, extra) in gaps.items():
        role = "false_alarm" if case.startswith("fault_free") else "detection"
        for thr in cands:
            fl = stuck_flag.rolling_flag(gap, thr, args.window, args.min_rows)
            share = np.nan
            if extra is not None:
                df, pos = extra
                cmd = df.loc[gap.index, "OA_DMPR_DM"]
                share = float(((cmd - pos).abs() > thr).mean())
            row = rates_row(case, split, role, fl, thr, share)
            sweep_rows.append(row)
            if thr == threshold:
                rate_rows.append(row)
        fl = stuck_flag.rolling_flag(gap, threshold, args.window, args.min_rows)
        if role == "detection":
            d = fl[fl["decided"]]
            for month, g in d["flag"].groupby(d.index.month):
                month_rows.append({"case": case, "month": int(month), "n_decided": int(len(g)),
                                   "detection_row_rate": float(g.mean())})
    if threshold not in cands:
        print("[note] chosen threshold is not in the candidate list; sweep omits it")
    rates_df, sweep_df, month_df = map(pd.DataFrame, (rate_rows, sweep_rows, month_rows))

    cmd_check = float((test[col] - test["OA_DMPR_DM"]).abs().mean())
    settings = pd.DataFrame({"setting": ["estimator", "variant", "rule", "knots", "window", "min_rows",
                                         "threshold", "threshold_chosen_by", "max_false_alarm",
                                         "fault_free_test_mean_abs_actual_minus_command"],
                             "value": [est.name, args.variant, args.rule, args.knots, args.window,
                                       args.min_rows, threshold, how, args.max_false_alarm, cmd_check]})

    t = config.TABLES_DIR
    rates_df.to_csv(t / "stuck_flag_rates.csv", index=False)
    sweep_df.to_csv(t / "stuck_flag_sweep.csv", index=False)
    month_df.to_csv(t / "stuck_flag_by_month.csv", index=False)
    settings.to_csv(t / "stuck_flag_settings.csv", index=False)
    make_tradeoff_figure(sweep_df, threshold, f"Stuck-damper flag: {est.name}, window {args.window}").savefig(
        config.FIGURES_DIR / "stuck_flag_tradeoff.png", dpi=150)

    # ---- console summary
    fmt = lambda v: f"{v:.3g}"
    show = rates_df[["case", "role", "n_decided", "coverage", "row_flag_rate", "day_flag_rate",
                     "miss_row_rate", "detectable_share"]]
    print(f"\nStuck-damper flag with {est.name}, threshold {threshold:g}, window {args.window}")
    print("false_alarm rows: share of fault-free TEST rows flagged (want near 0)")
    print("detection rows:   share of stuck-file rows flagged (want near 1); miss = 1 - detection")
    print(show.to_string(index=False, float_format=fmt))
    print(f"\nCheck: mean |actual - command| on fault-free test rows = {cmd_check:.3g} damper units "
          "(small means the command is a fair stand-in for the true position when nothing is wrong).")
    print("\nReminders: the threshold was chosen on fault-free validation rows only. The stuck files use "
          "different weather from the fault-free run, and the false-alarm rate is from one simulated year.")
    print(f"Tables written to {t}\nFigures written to {config.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
