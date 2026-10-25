"""Runtime, scaling, profile and memory of every damper estimator (report section 6.2).

Run from the project root, after the audit:

    python -m scripts.run_timing
    python -m scripts.run_timing --repeats 10 --sizes 10000 100000 1000000
    python -m scripts.run_timing --rule quantile --knots 45

Methods timed (all fitted on the fault-free TRAIN block, tuned on the VALIDATION block):
  linear_inverted   baseline in the report template: f = a + b*position, inverted
  direct_poly2      position = polynomial in f
  logistic_curve    two-parameter physical curve
  cubic_spline      natural cubic spline through binned medians
  monotone_spline   isotonic knots + PCHIP

For each method:
  fit_s         time to fit once with the settings fixed (median and best of ``--repeats``)
  tune_s        time of the whole tuning step (knot count or fit range chosen on validation, then
                the final fit). The linear and polynomial models have nothing to tune.
  predict_s_1M  time to predict one million rows (the cost of using the model)
  predict_1_ms  time of one single-row predict call (a streaming deployment calls it row by row)
  fit_mb, predict_mb   peak extra memory from ``tracemalloc``
  ops_estimate  estimated operation count for the train size (formulae in ``OPS``; they are
                order-of-magnitude estimates, not counted instructions)
  exponent      measured scaling: slope of log(fit time) against log(rows). 1.0 is linear in n.

Timing uses ``time.perf_counter`` after one warm-up call. Times depend on the machine, so
report the machine alongside them (printed at the top). Scaling rows beyond the real train size
are drawn with replacement, which is fine for timing but not for accuracy.

Outputs (under ``results/``)
  tables/timing_summary.csv      one row per method (the table for section 6.2)
  tables/timing_scaling.csv      fit time against number of training rows, per method
  tables/timing_tuning.csv       time of each knot-count candidate and logistic fit candidate
  figures/timing_scaling.png     fit time against rows (log-log)
  tables/profile_<method>.txt    cProfile, top 15 by cumulative time, of the tuning step
"""
from __future__ import annotations

import argparse
import cProfile
import io
import platform
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import config
from ahu_oa import damper_model, parametric_model, spline_model, splits
from scripts.run_damper_model import prepare

METHODS = ("linear_inverted", "direct_poly2", "logistic_curve", "cubic_spline", "monotone_spline")

# Order-of-magnitude operation counts for n training rows, k knots, C candidate settings,
# m predicted rows. These are stated assumptions, not measurements.
OPS = {
    "linear_inverted": ("~6n: two means, a covariance and a variance in one pass; predict ~2m",
                        lambda n, k, C, m: 6 * n),
    "direct_poly2": ("~15n: normal equations for a degree-2 polynomial; predict ~4m",
                     lambda n, k, C, m: 15 * n),
    "logistic_curve": ("~10n: logit of f (3 transcendental ops) plus a straight-line fit; predict ~6m",
                       lambda n, k, C, m: 10 * n),
    "cubic_spline": ("n*log2(k) to bin f, ~n*log2(n/k) for the per-bin medians, O(k) to build the "
                     "spline; predict m*log2(k)",
                     lambda n, k, C, m: n * np.log2(max(k, 2)) + n * np.log2(max(n / max(k, 1), 2))),
    "monotone_spline": ("as the cubic spline plus O(k) isotonic pass (pool-adjacent-violators); "
                        "predict m*log2(k)",
                        lambda n, k, C, m: n * np.log2(max(k, 2)) + n * np.log2(max(n / max(k, 1), 2)) + 4 * k),
}


def time_call(fn, repeats: int) -> tuple[float, float]:
    """Median and best wall time of ``fn()`` over ``repeats`` calls, after one warm-up."""
    fn()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t)), float(np.min(t))


def peak_mb(fn) -> float:
    """Peak extra memory (MB) allocated while ``fn()`` runs, from ``tracemalloc``."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1e6


def make_methods(col: str, degree: int, rule: str, knots: int, candidates, tolerance: float,
                 abs_tolerance: float) -> dict:
    """Map method name to ``(fit_fn, tune_fn)``; each takes ``(train, val)``.

    ``fit_fn`` fits once with fixed settings; ``tune_fn`` includes the tuning step
    and returns the final estimator (None where there is nothing to tune).
    """
    def spline_tune(mono):
        def run(train, val):
            k, _ = spline_model.select_n_knots(train, val, col, candidates=tuple(candidates), rule=rule,
                                               tolerance=tolerance, monotone=mono, abs_tolerance=abs_tolerance)
            return spline_model.fit_spline(train, col, n_knots=k, rule=rule, monotone=mono)
        return run

    return {
        "linear_inverted": (lambda tr, va: damper_model.fit_inverted(tr, col), None),
        "direct_poly2": (lambda tr, va: damper_model.fit_direct(tr, col, degree), None),
        "logistic_curve": (lambda tr, va: parametric_model.fit_logistic(tr, col),
                           lambda tr, va: parametric_model.select_logistic(tr, va, col)[0]),
        "cubic_spline": (lambda tr, va: spline_model.fit_spline(tr, col, n_knots=knots, rule=rule),
                         spline_tune(False)),
        "monotone_spline": (lambda tr, va: spline_model.fit_spline(tr, col, n_knots=knots, rule=rule,
                                                                   monotone=True),
                            spline_tune(True)),
    }


def profile_text(fn, top: int = 15) -> str:
    """cProfile of ``fn()``: top ``top`` entries by cumulative time, as text."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def make_scaling_figure(scaling: pd.DataFrame) -> Figure:
    """Fit time against number of training rows, log-log, one line per method."""
    fig = Figure(figsize=(7, 4.8))
    ax = fig.subplots()
    for name, g in scaling.groupby("method", sort=False):
        ax.plot(g["n_rows"], g["fit_s"], marker="o", label=name)
    n = np.array(sorted(scaling["n_rows"].unique()), dtype=float)
    if len(n) > 1:
        ref = scaling["fit_s"].min() * n / n[0]
        ax.plot(n, ref, color="grey", linestyle=":", label="linear in n (reference)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training rows n")
    ax.set_ylabel("Fit time (s, median)")
    ax.set_title("Fit time against training rows")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    parser.add_argument("--settle", default=config.SETTLE)
    parser.add_argument("--degree", type=int, default=config.DIRECT_POLY_DEGREE)
    parser.add_argument("--repeats", type=int, default=5, help="timed repeats per measurement")
    parser.add_argument("--rule", default="uniform", choices=list(spline_model.KNOT_RULES))
    parser.add_argument("--knots", type=int, default=45, help="fixed knot count for the fit-only timing")
    parser.add_argument("--candidates", nargs="+", type=int,
                        default=list(spline_model.DEFAULT_KNOT_CANDIDATES),
                        help="knot counts tried by the tuning step")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--abs-tolerance", type=float, default=spline_model.DEFAULT_ABS_TOLERANCE)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10_000, 30_000, 100_000, 300_000],
                        help="training-row counts for the scaling test")
    parser.add_argument("--predict-rows", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    col = config.DAMPER_COL
    path = args.data_dir / config.FILES["fault_free"]
    if not path.exists():
        print(f"[problem] fault-free file {path} not found.")
        return 1

    print(f"Machine: {platform.platform()}, {platform.processor() or 'cpu n/a'}, "
          f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}")
    t0 = time.perf_counter()
    ff, report = prepare(path, args.min_delta, args.settle)
    t_prepare = time.perf_counter() - t0
    parts = splits.split_data(ff, method=config.SPLIT_METHOD)
    train, val = parts["train"], parts["val"]
    n_train = int(train["well_conditioned"].sum())
    print(f"[ok] load + filter + f: {t_prepare:.2f} s for {report['n_input']} raw rows "
          f"({report['n_complete']} kept); train {len(train)} rows ({n_train} well-conditioned)")

    methods = make_methods(col, args.degree, args.rule, args.knots, args.candidates,
                           args.tolerance, args.abs_tolerance)
    f_big = train["f_phys"].dropna().sample(args.predict_rows, replace=True, random_state=0).to_numpy()
    f_one = f_big[:1]

    rows, tuned = [], {}
    for name in METHODS:
        fit_fn, tune_fn = methods[name]
        est = fit_fn(train, val)
        fit_med, fit_min = time_call(lambda: fit_fn(train, val), args.repeats)
        if tune_fn is not None:
            tune_med, tune_min = time_call(lambda: tune_fn(train, val), max(1, min(args.repeats, 3)))
            tuned[name] = tune_fn(train, val)
        else:
            tune_med = tune_min = np.nan
        final = tuned.get(name, est)
        pred_med, _ = time_call(lambda: final.predict(f_big), args.repeats)
        one_med, _ = time_call(lambda: final.predict(f_one), 200)
        k = len(final.params["x"]) if "x" in final.params else 0
        n_cand = len(args.candidates) if name.endswith("spline") else 1
        formula, ops_fn = OPS[name]
        rows.append({
            "method": name, "n_train": n_train, "fit_s": fit_med, "fit_s_best": fit_min,
            "tune_s": tune_med, "predict_s_1M": pred_med * 1e6 / len(f_big),
            "predict_1_ms": one_med * 1e3,
            "fit_mb": peak_mb(lambda: fit_fn(train, val)),
            "predict_mb": peak_mb(lambda: final.predict(f_big)),
            "knots_final": k,
            "ops_fit_estimate": float(ops_fn(n_train, k or args.knots, n_cand, len(f_big))),
            "ops_formula": formula})
        print(f"[ok] {name}: fit {fit_med * 1e3:.1f} ms, tune "
              f"{'-' if np.isnan(tune_med) else f'{tune_med:.2f} s'}, predict 1M {pred_med:.3f} s")

    # ---- scaling with the number of training rows (fixed settings, fit only)
    scaling = []
    for n in args.sizes:
        sub = train[train["well_conditioned"]].sample(n, replace=n > n_train, random_state=1)
        for name in METHODS:
            fit_fn = methods[name][0]
            med, _ = time_call(lambda: fit_fn(sub, val), max(1, min(args.repeats, 3)))
            scaling.append({"method": name, "n_rows": n, "fit_s": med})
    scaling_df = pd.DataFrame(scaling)
    exps = {name: float(np.polyfit(np.log(g["n_rows"]), np.log(g["fit_s"]), 1)[0])
            for name, g in scaling_df.groupby("method")} if len(args.sizes) > 1 else {}
    summary = pd.DataFrame(rows)
    summary["exponent"] = summary["method"].map(exps)

    # ---- cost of each tuning candidate
    cand_rows = []
    for mono in (False, True):
        for k in args.candidates:
            try:
                med, _ = time_call(lambda: spline_model.fit_spline(train, col, n_knots=k, rule=args.rule,
                                                                   monotone=mono), 3)
            except ValueError:
                continue
            cand_rows.append({"method": "monotone_spline" if mono else "cubic_spline",
                              "candidate": f"{k} knots", "fit_s": med})
    _, lsel = parametric_model.select_logistic(train, val, col)
    for r in lsel.itertuples():
        lo, hi = (r.low, r.high)
        med, _ = time_call(lambda: parametric_model.fit_logistic(
            train, col, fit_range=(lo, hi) if r.method == "inverse" else parametric_model.FIT_RANGE,
            method=r.method), 3)
        cand_rows.append({"method": "logistic_curve",
                          "candidate": f"{r.method} {lo:g}-{hi:g}" if r.method == "inverse" else "forward",
                          "fit_s": med})

    t = config.TABLES_DIR
    summary.to_csv(t / "timing_summary.csv", index=False)
    scaling_df.to_csv(t / "timing_scaling.csv", index=False)
    pd.DataFrame(cand_rows).to_csv(t / "timing_tuning.csv", index=False)
    make_scaling_figure(scaling_df).savefig(config.FIGURES_DIR / "timing_scaling.png", dpi=150)
    for name in METHODS:
        fit_fn, tune_fn = methods[name]
        fn = (lambda: tune_fn(train, val)) if tune_fn else (lambda: fit_fn(train, val))
        (t / f"profile_{name}.txt").write_text(profile_text(fn))

    fmt = lambda v: f"{v:.4g}"
    show = summary[["method", "fit_s", "tune_s", "predict_s_1M", "predict_1_ms", "fit_mb",
                    "predict_mb", "ops_fit_estimate", "exponent"]]
    print("\nTiming summary (seconds unless stated; fit_mb and predict_mb are peak extra MB):")
    print(show.to_string(index=False, float_format=fmt))
    print("\nOperation-count assumptions:")
    for name in METHODS:
        print(f"  {name}: {OPS[name][0]}")
    print(f"\nTables written to {t}\nFigure written to {config.FIGURES_DIR / 'timing_scaling.png'}")
    print("Reminders: times depend on this machine; a quantile rule costs more than uniform; "
          "scaling rows above the real train size are resampled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
