"""Fit and validate the spline damper estimators: position = g(f).

Baseline: the natural cubic spline. Alternative: the monotone spline (isotonic regression
on the knot positions, then PCHIP), whose estimated position can never fall as ``f`` rises
(see ``ahu_oa.spline_model``). Both are compared on identical data and splits.

Run from the project root (audit first, see ``scripts/run_damper_audit.py``):

    python -m scripts.run_spline_model
    python -m scripts.run_spline_model --rules uniform
    python -m scripts.run_spline_model --abs-tolerance 0.001 --tolerance 0.05
    python -m scripts.run_spline_model --abs-tolerance 0 --tolerance 0     # plain lowest error

What it does
  1. Loads the fault-free file (fan on, settled, complete rows), computes the
     outdoor air fraction f from the three temperatures, and splits the rows
     into train / validation / test blocks in time (``config.SPLIT_METHOD``).
  2. For each method (``cubic`` baseline, ``monotone`` alternative) and knot rule
     (``quantile`` = equal rows per bin, ``uniform`` = equal width bins) chooses the
     number of knots on the VALIDATION block, then fits the spline on the TRAIN block.
     The knot count is NOT the absolute minimum of the validation error: it is the
     smallest local minimum of the error curve whose error is within ``--tolerance``
     (relative) or ``--abs-tolerance`` (damper units) of the best. This avoids paying for
     a very large knot count when a much smaller one is almost as accurate.
  3. For each method, keeps the rule with the lowest validation MAE as that method's final.
  4. Scores every fitted spline on the validation block, the test block, and every
     stuck-damper file against its known position, and writes a baseline-versus-alternative
     comparison.

Honest-validation notes (also printed at the end)
  * The validation block is used to choose the knot count and the rule, so the
    validation error is optimistic for the splines. The test block and the
    stuck files are never used for any choice: report those as the real result.
  * The stuck files come from separate runs with different weather, so some of
    any error there is weather, not model.

Outputs (under ``results/``)
  tables/spline_knot_selection.csv     validation MAE for every knot candidate, variant and rule
  tables/spline_selected.csv           chosen knots per variant and rule, monotone check, final flag
  tables/spline_knots.csv              the fitted knot points (f, position, rows behind each)
  tables/spline_comparison.csv         one row per estimator: test MAE and stuck-file errors
  tables/spline_model_metrics.csv      all estimators x val / test / stuck x conditioning
  tables/spline_model_stuck_summary.csv, spline_model_stuck_by_month.csv
  tables/spline_model_error_by_bin.csv test-block error by true damper position
  figures/spline_curve.png             f vs position with every fitted curve and the stuck points
  figures/spline_curve_low_f.png       the same, zoomed on small f where the curves differ most
  figures/spline_knot_selection.png    validation MAE against knot count
  figures/spline_fault_free_test.png   estimated vs true position, test block
  figures/spline_stuck_<case>.png      implied-position histogram per stuck file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import config
from ahu_oa import spline_model, splits
from scripts.run_damper_model import (make_fault_free_figure, make_stuck_figure,
                                      prepare, score_estimator, stuck_detail)

BIN_WIDTH = 0.05
VARIANTS = ("cubic", "monotone")      # baseline, alternative
BASELINE, ALTERNATIVE = VARIANTS


def monotone_check(est, n: int = 4000) -> dict:
    """Check whether a fitted curve ever decreases as ``f`` increases.

    A cubic spline through noisy knot points can wiggle, but physically a more
    open damper cannot let less outdoor air in, so the curve should not fall.

    Args:
        est: Fitted estimator with a ``params["x"]`` knot list and ``predict``.
        n: Number of grid points between the first and last knot.

    Returns:
        ``{"monotone": bool, "max_decrease": largest drop in position between
        neighbouring grid points, "share_decreasing": share of grid steps that fall}``.
    """
    x = np.asarray(est.params["x"], dtype=float)
    pred = est.predict(np.linspace(x[0], x[-1], n))
    step = np.diff(pred)
    return {"monotone": bool((step >= -1e-9).all()),
            "max_decrease": float(max(0.0, -step.min())),
            "share_decreasing": float((step < -1e-9).mean())}


def error_by_bin(block: pd.DataFrame, estimators, col: str, width: float = BIN_WIDTH) -> pd.DataFrame:
    """Error by true damper position on the well-conditioned rows of one block.

    Args:
        block: Dataframe with ``f_phys``, ``well_conditioned`` and ``col``.
        estimators: Fitted estimators (anything with ``name`` and ``predict``).
        col: Column holding the true damper position.
        width: Bin width in damper units.

    Returns:
        Long dataframe: estimator, bin_low, bin_high, n, mae, bias.
    """
    wc = block[block["well_conditioned"]]
    truth = wc[col].to_numpy(float)
    edges = np.arange(0.0, 1.0 + width / 2, width)
    cut = pd.cut(truth, edges, include_lowest=True)
    rows = []
    for e in estimators:
        err = e.predict(wc["f_phys"].to_numpy(float)) - truth
        g = pd.DataFrame({"bin": cut, "err": err}).dropna().groupby("bin", observed=True)["err"]
        for interval, s in g:
            rows.append({"estimator": e.name, "bin_low": float(interval.left),
                         "bin_high": float(interval.right), "n": int(s.size),
                         "mae": float(s.abs().mean()), "bias": float(s.mean())})
    return pd.DataFrame(rows)


def make_curve_figure(train: pd.DataFrame, estimators, final_names,
                      stuck_points, col: str) -> Figure:
    """Fault-free training data (f on x, position on y) with every fitted curve.

    The stuck-file points are (median f, known position): a point far from the
    fault-free cloud shows where a model trained on fault-free data cannot work.
    """
    wc = train[train["well_conditioned"]][["f_phys", col]].dropna()
    f_lo, f_hi = np.percentile(wc["f_phys"], [0.5, 99.5])
    if stuck_points:
        f_lo = min(f_lo, min(p[2] for p in stuck_points))
        f_hi = max(f_hi, max(p[2] for p in stuck_points))
    f_lo, f_hi = f_lo - 0.02, f_hi + 0.02
    fig = Figure(figsize=(9, 6.2))
    ax = fig.subplots()
    hb = ax.hexbin(wc["f_phys"], wc[col], gridsize=70, mincnt=1, bins="log", cmap="Greys",
                   extent=(f_lo, f_hi, -0.02, 1.02))
    fig.colorbar(hb, ax=ax, label="fault-free training rows (log scale)")
    grid = np.linspace(f_lo, f_hi, 500)
    styles = ["--", ":", "-", "-."]
    for i, e in enumerate(estimators):
        final = e.name in final_names
        ax.plot(grid, e.predict(grid), styles[i % len(styles)], linewidth=2.6 if final else 1.4,
                label=e.name + (" (final)" if final else ""))
        if e.kind == "spline" and final:
            ax.plot(e.params["x"], e.params["y"], "o", markersize=3, label=f"knots of {e.name}")
    for case, pos, f_med in stuck_points:
        ax.plot(f_med, pos, "*", color="red", markersize=14, markeredgecolor="k")
        ax.annotate(f"stuck {pos:g}", (f_med, pos), textcoords="offset points", xytext=(6, -12),
                    color="red", fontsize=9)
    ax.plot([], [], "*", color="red", markersize=12, markeredgecolor="k",
            label="stuck file: (median f, known position)")
    ax.set_xlim(f_lo, f_hi)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Outdoor air fraction f (calculated)")
    ax.set_ylabel(f"Damper position ({col})")
    ax.set_title("Fault-free training data and fitted f-to-position curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def make_zoom_figure(train: pd.DataFrame, estimators, final_names, col: str,
                     f_max: float = 0.25) -> Figure:
    """Fault-free training data and every curve for small ``f`` only.

    Small fractions are where a cubic spline can dip or chase the transient rows (damper
    at its minimum while ``f`` is still high), so this is where the monotone spline
    differs most.
    """
    wc = train[train["well_conditioned"] & (train["f_phys"] <= f_max)][["f_phys", col]].dropna()
    fig = Figure(figsize=(9, 5.4))
    ax = fig.subplots()
    hb = ax.hexbin(wc["f_phys"], wc[col], gridsize=60, mincnt=1, bins="log", cmap="Greys",
                   extent=(-0.01, f_max, -0.02, 1.02))
    fig.colorbar(hb, ax=ax, label="fault-free training rows (log scale)")
    grid = np.linspace(-0.01, f_max, 600)
    for e in estimators:
        if e.kind != "spline":
            continue
        final = e.name in final_names
        ax.plot(grid, e.predict(grid), "-" if e.params.get("monotone") else "--",
                linewidth=2.6 if final else 1.3, label=e.name + (" (final)" if final else ""))
    ax.set_xlim(-0.01, f_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Outdoor air fraction f (calculated)")
    ax.set_ylabel(f"Damper position ({col})")
    ax.set_title("Small-f region: cubic (dashed) and monotone (solid) splines")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def comparison_table(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, selected_df: pd.DataFrame,
                     estimators, stuck_cases) -> pd.DataFrame:
    """One readable row per estimator: held-out test error and error on each stuck file.

    Args:
        metrics_df: Long metrics table (``group``, ``split``, ``case``, ``estimator``, ``mae``).
        summary_df: Stuck-file summary (``case``, ``estimator``, ``implied_median``).
        selected_df: Chosen-knot table with the ``estimator`` and ``monotone`` columns.
        estimators: Fitted estimators, in display order.
        stuck_cases: Stuck-case labels to include as columns.

    Returns:
        Dataframe with ``estimator``, ``monotone`` (curve passes the check), ``test_mae``, ``<case>_mae`` and ``<case>_median`` for each case.
        MAE is in damper units on well-conditioned rows; ``test_mae`` is the honest
        fault-free number (the validation block was used for tuning).
    """
    wc = metrics_df[metrics_df["group"] == "well_conditioned"]
    mono = dict(zip(selected_df["estimator"], selected_df["monotone"]))
    rows = []
    for e in estimators:
        row = {"estimator": e.name, "monotone": mono.get(e.name, np.nan)}
        t = wc[(wc["estimator"] == e.name) & (wc["split"] == "test")]
        row["test_mae"] = float(t["mae"].iloc[0]) if len(t) else np.nan
        for case in stuck_cases:
            m = wc[(wc["estimator"] == e.name) & (wc["case"] == case)]
            row[f"{case}_mae"] = float(m["mae"].iloc[0]) if len(m) else np.nan
            if not summary_df.empty:
                sm = summary_df[(summary_df["estimator"] == e.name) & (summary_df["case"] == case)]
                row[f"{case}_median"] = float(sm["implied_median"].iloc[0]) if len(sm) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_knot_figure(sel: pd.DataFrame) -> Figure:
    """Validation MAE against knot count, one line per variant and rule, chosen count ringed."""
    fig = Figure(figsize=(7, 4.6))
    ax = fig.subplots()
    sel = sel.assign(label=sel["variant"] + " " + sel["rule"])
    for label, g in sel.groupby("label"):
        line, = ax.plot(g["n_knots"], g["val_mae"], marker="o", label=f"{label} knots")
        c = g[g["chosen"]]
        ax.plot(c["n_knots"], c["val_mae"], "o", markersize=14, markerfacecolor="none",
                markeredgecolor=line.get_color(), markeredgewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of knots (bins)")
    ax.set_ylabel("Validation MAE (damper units)")
    ax.set_title("Knot count chosen on the validation block (ringed)")
    ax.legend()
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    parser.add_argument("--settle", default=config.SETTLE)
    parser.add_argument("--tol", type=float, default=config.POSITION_TOL,
                        help="stuck-file agreement tolerance (damper units)")
    parser.add_argument("--rules", nargs="+", default=list(spline_model.KNOT_RULES),
                        choices=list(spline_model.KNOT_RULES), help="knot rules to compare")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                        help="spline variants to fit: cubic (natural cubic spline) and/or "
                             "monotone (isotonic knots + PCHIP)")
    parser.add_argument("--show-knots", nargs="+", type=int, default=[45, 250],
                        help="fixed uniform-knot counts drawn on the curve figures for both "
                             "the cubic and the monotone spline")
    parser.add_argument("--figure-knots", type=int, default=45,
                        help="uniform-knot count of the cubic and monotone splines used for the "
                             "fault-free test and stuck-file figures")
    parser.add_argument("--knots", nargs="+", type=int,
                        default=list(spline_model.DEFAULT_KNOT_CANDIDATES),
                        help="knot counts to try")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="knot choice: a local minimum is good enough if its val MAE is "
                             "within this share of the best (relative margin)")
    parser.add_argument("--abs-tolerance", type=float, default=spline_model.DEFAULT_ABS_TOLERANCE,
                        help="knot choice: ... or within this many damper units of the best "
                             "(the larger of the two margins applies)")
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
    print(f"[ok] fault_free: {report['n_complete']} fan-on settled rows, "
          f"{100 * report['coverage_well_conditioned']:.1f}% well-conditioned")

    parts = splits.split_data(ff, method=config.SPLIT_METHOD)
    train, val, test = parts["train"], parts["val"], parts["test"]
    print(f"[ok] split ({config.SPLIT_METHOD}): train {len(train)}, val {len(val)}, test {len(test)} rows")

    # ---- baseline (cubic) and alternative (monotone) splines, fitted on the train block only
    splines, sel_tables, selected = {}, [], []
    for variant in args.variants:
        for rule in args.rules:
            mono = variant == ALTERNATIVE
            k, table = spline_model.select_n_knots(
                train, val, col, candidates=tuple(args.knots), rule=rule,
                tolerance=args.tolerance, monotone=mono, abs_tolerance=args.abs_tolerance)
            est = spline_model.fit_spline(train, col, n_knots=k, rule=rule, monotone=mono)
            sel_tables.append(table.assign(variant=variant, rule=rule,
                                           chosen=table["n_knots"] == k))
            val_mae = float(table.loc[table["n_knots"] == k, "val_mae"].iloc[0])
            best_row = table.loc[table["val_mae"].idxmin()]
            minima = table.loc[table["local_min"], "n_knots"].tolist()
            selected.append({"variant": variant, "rule": rule, "estimator": est.name,
                             "n_knots_chosen": k, "n_knots_fitted": len(est.params["x"]),
                             "val_mae": val_mae,
                             "lowest_val_mae_knots": int(best_row["n_knots"]),
                             "lowest_val_mae": float(best_row["val_mae"]),
                             "val_mae_cost": val_mae - float(best_row["val_mae"]),
                             "local_minima": " ".join(str(m) for m in minima),
                             **monotone_check(est)})
            splines[(variant, rule)] = est
    sel_df = pd.concat(sel_tables, ignore_index=True)
    selected_df = pd.DataFrame(selected)
    # one final spline per method: the rule with the lowest validation MAE
    finals = {}
    for variant in args.variants:
        g = selected_df[selected_df["variant"] == variant]
        finals[variant] = g["val_mae"].idxmin()
    selected_df["final"] = selected_df.index.isin(list(finals.values()))
    final_est = {v: splines[(selected_df.loc[i, "variant"], selected_df.loc[i, "rule"])]
                 for v, i in finals.items()}
    # fixed uniform-knot splines (e.g. 45 and 250) for the curve figures, fitted on train only
    fixed = []
    for v in args.variants:
        for k in args.show_knots:
            fixed.append(spline_model.fit_spline(train, col, n_knots=k, rule="uniform",
                                                 monotone=v == ALTERNATIVE))
    # the splines drawn in the fault-free test and stuck-file figures: uniform knots, fixed count
    figure_est = {v: spline_model.fit_spline(train, col, n_knots=args.figure_knots, rule="uniform",
                                             monotone=v == ALTERNATIVE) for v in args.variants}
    estimators = list(splines.values()) + [e for e in fixed if e.name not in
                                           {s_.name for s_ in splines.values()}]
    final_list = [final_est[v] for v in args.variants]
    spline_names = {e.name for e in estimators}

    # ---- fault-free validation and test blocks
    metric_rows = []
    for split in ("val", "test"):
        block = parts[split]
        for e in estimators:
            metric_rows += score_estimator("fault_free", split, e, block[col].to_numpy(float), block)

    # ---- stuck files (never used for any choice)
    summary_rows, month_rows, stuck_points = [], [], []
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
        f_wc = df.loc[df["well_conditioned"], "f_phys"].dropna()
        if len(f_wc):
            stuck_points.append((case, float(position), float(f_wc.median())))
        for v, e in figure_est.items():
            make_stuck_figure(case, e, df, position, args.min_delta).savefig(
                config.FIGURES_DIR / f"spline_stuck_{case}_{v}.png", dpi=150)
        print(f"[ok] {case}: {rep['n_complete']} fan-on settled rows")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["tuned_on_this_block"] = (metrics_df["split"] == "val") & metrics_df["estimator"].isin(spline_names)
    summary_df, month_df = pd.DataFrame(summary_rows), pd.DataFrame(month_rows)
    bins_df = error_by_bin(test, estimators, col)

    # ---- write everything
    t = config.TABLES_DIR
    sel_df.to_csv(t / "spline_knot_selection.csv", index=False)
    selected_df.to_csv(t / "spline_selected.csv", index=False)
    pd.concat([pd.DataFrame({"estimator": e.name, "variant": v, "rule": r, "f": e.params["x"],
                             "position": e.params["y"], "rows": e.params["rows"]})
               for (v, r), e in splines.items()], ignore_index=True).to_csv(t / "spline_knots.csv", index=False)
    metrics_df.to_csv(t / "spline_model_metrics.csv", index=False)
    summary_df.to_csv(t / "spline_model_stuck_summary.csv", index=False)
    month_df.to_csv(t / "spline_model_stuck_by_month.csv", index=False)
    bins_df.to_csv(t / "spline_model_error_by_bin.csv", index=False)
    final_names = set()          # curve figures show the fixed uniform splines, none emphasised
    make_curve_figure(train, fixed, final_names, stuck_points, col).savefig(
        config.FIGURES_DIR / "spline_curve.png", dpi=150)
    make_zoom_figure(train, fixed, final_names, col).savefig(
        config.FIGURES_DIR / "spline_curve_low_f.png", dpi=150)
    make_knot_figure(sel_df).savefig(config.FIGURES_DIR / "spline_knot_selection.png", dpi=150)
    stuck_cases = [c for c in config.STUCK_FILES if c in set(metrics_df["case"])]
    compare_df = comparison_table(metrics_df, summary_df, selected_df, estimators, stuck_cases)
    compare_df.to_csv(t / "spline_comparison.csv", index=False)
    for v, e in figure_est.items():
        make_fault_free_figure(e, test, col).savefig(
            config.FIGURES_DIR / f"spline_fault_free_test_{v}.png", dpi=150)

    # ---- console summary
    fmt = lambda v: f"{v:.4g}"
    print("\nKnot selection (validation block). lowest_val_mae_knots is the absolute minimum; "
          "n_knots_chosen is the smallest local minimum within tolerance:")
    show_sel = selected_df[["variant", "rule", "n_knots_chosen", "val_mae", "lowest_val_mae_knots",
                            "lowest_val_mae", "val_mae_cost", "local_minima", "monotone", "final"]]
    print(show_sel.to_string(index=False, float_format=fmt))
    for v in args.variants:
        role = "Baseline" if v == BASELINE else "Alternative"
        print(f"\n{role} final spline: {final_est[v].name}  (lowest validation MAE of the rules tried)")

    wcm = metrics_df[metrics_df["group"] == "well_conditioned"].copy()
    wcm["where"] = np.where(wcm["split"] == "stuck", wcm["case"], wcm["split"])
    order = ["val", "test"] + [c for c in config.STUCK_FILES if c in set(wcm["case"])]
    mae = wcm.pivot(index="estimator", columns="where", values="mae")
    mae = mae[[c for c in order if c in mae.columns]].reindex([e.name for e in estimators])
    print("\nMAE in damper units (well-conditioned rows). val = tuned on for splines; "
          "test and stuck are the honest numbers:")
    print(mae.to_string(float_format=fmt))
    if not summary_df.empty:
        med = summary_df.pivot(index="estimator", columns="case", values="implied_median")
        med = med[[c for c in config.STUCK_FILES if c in med.columns]].reindex([e.name for e in estimators])
        known = "  ".join(f"{c.split('_')[-1]}={config.STUCK_POSITION[c]:g}" for c in med.columns)
        print(f"\nMedian implied position on each stuck file (known: {known}):")
        print(med.to_string(float_format=fmt))
    print("\nComparison (damper units; test_mae is the honest fault-free error, "
          "*_mae columns are errors on the stuck files):")
    show = compare_df[["estimator", "monotone", "test_mae"] + [f"{c}_mae" for c in stuck_cases]]
    print(show.to_string(index=False, float_format=fmt))
    bad = selected_df[~selected_df["monotone"]]
    if not bad.empty:
        print("\n[note] not monotone: " + ", ".join(
            f"{r.estimator} (largest drop {r.max_decrease:.3g})" for r in bad.itertuples()))
    print("\nReminders: the validation block chose the knots and the rule; the stuck files use "
          "different weather from the fault-free run.")
    print(f"Tables written to {t}\nFigures written to {config.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
