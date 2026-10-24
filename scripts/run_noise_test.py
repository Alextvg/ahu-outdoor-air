"""Noise sensitivity test for the cubic (baseline) and monotone (alternative) splines.

The model input f comes from three temperature sensors:
    f = (T_mixed - T_return) / (T_outdoor - T_return)
so sensor noise becomes noise in f, and from there error in the damper estimate. This script
adds extra Gaussian noise (standard deviation sigma, in K, independent for each of the three
temperatures) and measures how the damper error grows. The data already carry whatever noise
the simulation put in, so sigma is ADDED noise on top of that, and sigma = 0 is the
unmodified data.

Three experiments are run, each for the linear baseline and both splines (fixed knot count):
  test_noise   Fit on clean train rows, add noise to the test block only.
               Question: how does the finished model cope with noisy sensors in use?
  train_noise  Fit on noisy train rows, score on the clean test block.
               Question: how much does noise in the training data damage the fitted curve?
  both_noise   Noise in train and test at the same sigma (the realistic case).

Rows are judged well-conditioned from the NOISY temperatures (|T_outdoor - T_return| at least
``--min-delta``), because that is all a deployed model would see. Noise therefore also changes
which rows are answered, so coverage is reported next to the error. Each noisy setting is
repeated ``--repeats`` times with different random draws; the table gives mean and spread.

Run from the project root:

    python -m scripts.run_noise_test
    python -m scripts.run_noise_test --sigmas 0 0.25 0.5 1 2 --repeats 10 --knots 45

Limits: noise is independent between sensors and between rows, which is the mild case. Real
sensors also have offsets (calibration bias) and slow drift, which are not tested here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from ahu_oa import damper_model, metrics, physics, spline_model, splits
from scripts.run_damper_model import prepare

VARIANTS = ("linear", "cubic", "monotone")      # linear reference, cubic baseline, monotone alternative
EXPERIMENTS = ("test_noise", "train_noise", "both_noise")
TEMP_COLS = ("MA_TEMP_C", "RA_TEMP_C", "OA_TEMP_C")      # mixed, return, outdoor
# Title 24 (California Energy Code) requires economizer FDD temperature sensors accurate to
# +/-2 F over 40-80 F. A 2 F difference is 2 / 1.8 = 1.11 K. It is a tolerance limit, not a
# standard deviation, so it is drawn as a reference line rather than a result.
TITLE24_K = 2.0 / 1.8


def add_noise(df: pd.DataFrame, sigma: float, rng: np.random.Generator,
              min_delta: float) -> pd.DataFrame:
    """Return a copy with noisy temperatures and a recomputed ``f_phys`` and conditioning flag.

    Args:
        df: Dataframe with the three temperature columns in ``TEMP_COLS``.
        sigma: Standard deviation (K) of the Gaussian noise added to each temperature,
            drawn independently per sensor and per row. 0 leaves the temperatures as they are.
        rng: Random generator, so runs are repeatable.
        min_delta: Smallest ``|T_outdoor - T_return|`` (K) treated as reliable.

    Returns:
        Copy of ``df`` with noisy temperature columns, whose ``f_phys`` is NaN where the noisy gap is below ``min_delta``
        and whose ``well_conditioned`` is True exactly where ``f_phys`` is finite.
    """
    if sigma < 0:
        raise ValueError("sigma must not be negative")
    out = df.copy()
    noisy = {c: df[c].to_numpy(float) + (rng.normal(0.0, sigma, len(df)) if sigma > 0 else 0.0)
             for c in TEMP_COLS}
    for c in TEMP_COLS:                       # the linear baseline is fitted from the temperatures
        out[c] = noisy[c]
    out["f_phys"] = physics.oa_fraction_from_temps(
        noisy["MA_TEMP_C"], noisy["RA_TEMP_C"], noisy["OA_TEMP_C"], min_delta=min_delta)
    out["well_conditioned"] = np.isfinite(out["f_phys"].to_numpy(float))
    return out


def score(est, test: pd.DataFrame, col: str) -> dict:
    """MAE, RMSE, bias and coverage of one spline on one (possibly noisy) test block."""
    pred = est.predict(test["f_phys"].to_numpy(float))
    return metrics.error_summary(test[col].to_numpy(float), pred)


def fit_variant(variant: str, train: pd.DataFrame, col: str, n_knots: int, rule: str):
    """Fit one method: the linear reference or a (cubic / monotone) spline."""
    if variant == "linear":
        return damper_model.fit_inverted(train, col)
    return spline_model.fit_spline(train, col, n_knots=n_knots, rule=rule,
                                   monotone=variant == "monotone")


def run_experiments(train: pd.DataFrame, test: pd.DataFrame, col: str, sigmas, repeats: int,
                    n_knots: int, rule: str, min_delta: float, seed: int) -> pd.DataFrame:
    """Run every experiment, spline, sigma and repeat; return one row per run."""
    rng = np.random.default_rng(seed)
    clean_fit = {v: fit_variant(v, train, col, n_knots, rule) for v in VARIANTS}
    rows = []
    for sigma in sigmas:
        n_rep = 1 if sigma == 0 else repeats           # sigma 0 is deterministic
        for rep in range(n_rep):
            noisy_test = add_noise(test, sigma, rng, min_delta)
            noisy_train = add_noise(train, sigma, rng, min_delta)
            for v in VARIANTS:
                try:
                    noisy_fit = fit_variant(v, noisy_train, col, n_knots, rule)
                except ValueError:                     # too few usable bins at high noise
                    noisy_fit = None
                runs = {"test_noise": (clean_fit[v], noisy_test),
                        "train_noise": (noisy_fit, test),
                        "both_noise": (noisy_fit, noisy_test)}
                for name, (est, block) in runs.items():
                    if est is None:
                        continue
                    rows.append({"experiment": name, "variant": v, "sigma_K": sigma,
                                 "repeat": rep, **score(est, block, col)})
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over repeats for each experiment, spline and sigma."""
    g = runs.groupby(["experiment", "variant", "sigma_K"])
    out = g.agg(n_repeats=("mae", "size"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                rmse_mean=("rmse", "mean"), bias_mean=("bias", "mean"),
                coverage_mean=("coverage", "mean")).reset_index()
    out["mae_sd"] = out["mae_sd"].fillna(0.0)
    return out


def make_figure(summary: pd.DataFrame, n_knots: int) -> plt.Figure:
    """MAE against added noise for the three experiments, plus coverage."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    colour = {"linear": "tab:gray", "cubic": "tab:blue", "monotone": "tab:green"}
    style = {"linear": ("s", "-"), "cubic": ("o", "-"), "monotone": ("^", "--")}
    titles = {"test_noise": "Noise in test data only\n(model fitted on clean data)",
              "train_noise": "Noise in training data only\n(scored on clean test data)",
              "both_noise": "Noise in both"}
    for ax, name in zip(axes[:3], EXPERIMENTS):
        for v in VARIANTS:
            d = summary[(summary["experiment"] == name) & (summary["variant"] == v)]
            mk, ls = style[v]
            ax.errorbar(d["sigma_K"], d["mae_mean"], yerr=d["mae_sd"], marker=mk, linestyle=ls,
                        capsize=3, color=colour[v],
                        label="linear baseline" if v == "linear" else f"{v} spline")
        ax.set_title(titles[name], fontsize=10)
        ax.set_xlabel("added noise sigma per temperature (K)")
        ax.set_ylabel("test MAE (damper units)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    for ax in axes:
        ax.axvline(TITLE24_K, color="red", linestyle=":", linewidth=1.5,
                   label="Title 24 limit, ±2 °F = 1.1 K")
    axes[0].legend(fontsize=8)
    for v in VARIANTS:
        d = summary[(summary["experiment"] == "test_noise") & (summary["variant"] == v)]
        mk, ls = style[v]
        axes[3].plot(d["sigma_K"], d["coverage_mean"], marker=mk, linestyle=ls,
                     color=colour[v], label=v)
    axes[3].set_title("Share of test rows answered\n(noisy gap above the threshold)", fontsize=10)
    axes[3].set_xlabel("added noise sigma per temperature (K)")
    axes[3].set_ylabel("coverage")
    axes[3].set_ylim(0, 1.02)
    axes[3].grid(alpha=0.3)
    fig.suptitle(f"Noise sensitivity: linear baseline and {n_knots}-knot uniform splines", y=1.02)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    parser.add_argument("--settle", default=config.SETTLE)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0, 0.25, 0.5, 1.0, 2.0],
                        help="added noise levels (K), applied to each of the three temperatures")
    parser.add_argument("--repeats", type=int, default=5, help="random draws per noisy setting")
    parser.add_argument("--knots", type=int, default=45, help="number of knots")
    parser.add_argument("--rule", default="uniform", choices=list(spline_model.KNOT_RULES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = args.data_dir / config.FILES["fault_free"]
    if not path.exists():
        print(f"[problem] fault-free file {path} not found.")
        return 1
    try:
        ff, _ = prepare(path, args.min_delta, args.settle)
    except KeyError as err:
        print(f"[problem] column {err} not found. Run `python -m scripts.run_damper_audit`.")
        return 1
    parts = splits.split_data(ff, method=config.SPLIT_METHOD)
    train, test = parts["train"], parts["test"]
    print(f"[ok] train {len(train)}, test {len(test)} rows; {args.rule} knots, {args.knots}")

    runs = run_experiments(train, test, config.DAMPER_COL, sorted(set(args.sigmas)),
                           args.repeats, args.knots, args.rule, args.min_delta, args.seed)
    summary = summarise(runs)
    runs.to_csv(config.TABLES_DIR / "noise_test_runs.csv", index=False)
    summary.to_csv(config.TABLES_DIR / "noise_test_summary.csv", index=False)
    make_figure(summary, args.knots).savefig(config.FIGURES_DIR / "noise_test.png",
                                             dpi=150, bbox_inches="tight")

    show = summary.pivot_table(index=["experiment", "sigma_K"], columns="variant",
                               values=["mae_mean", "coverage_mean"])
    print("\nTest MAE and coverage by added noise (mean over repeats):")
    print(show.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote noise_test_summary.csv, noise_test_runs.csv and noise_test.png "
          f"to {config.TABLES_DIR.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
