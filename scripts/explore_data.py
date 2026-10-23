"""Explore the raw AHU CSV files and write ONE text report you can paste or attach.

Run from the project root:

    python -m scripts.explore_data
    python -m scripts.explore_data --data-dir path/to/csvs --min-delta 3

It works on the RAW column names and units (degF, CFM), so it does not depend on
``loading.py`` renaming anything, and it never assumes a column exists: missing
columns are reported instead of crashing.

The report (``results/data_report.txt``) answers, in order:
  0. Automatic flags: the things most likely to change the project plan.
  1. Files: rows, time range, step, duplicates, columns that differ between files.
  2. Every column in the fault-free file: unique values, missing, min/mean/max/std
     (constant columns are marked CONST).
  3. Occupancy: how SYS_CTL and SF_SPD_DM behave, and the weekly schedule.
  4. Mixing-equation fraction: how often it is well-conditioned, its range, by
     month, and which logged signals correlate with it (candidate ground truth).
  5. Bias files vs fault-free: which columns change, and is the OA_TEMP offset a
     constant?
  6. Detectability preview: a regression fitted on the first half of the
     fault-free year, and the residual it gives on the second half of each file.

Also saves ``results/tables/column_stats_fault_free.csv``.
Output uses plain ASCII so it prints cleanly in any Windows terminal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

F_TO_K = 5.0 / 9.0          # a temperature DIFFERENCE in degF times this is K
PLAUSIBLE_TOL = 0.05        # fraction counted as plausible within [-tol, 1+tol]
FEATURES = ["OA_DMPR_DM", "SF_CS"]   # signals used in the detectability preview


# --------------------------------------------------------------------------- helpers
def load_raw(path: Path):
    """Read a CSV with raw names, index it by time if possible. Returns (df, notes)."""
    notes = []
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    tcol = config.TIME_COL if config.TIME_COL in df.columns else df.columns[0]
    if tcol != config.TIME_COL:
        notes.append(f"config.TIME_COL '{config.TIME_COL}' not found; "
                     f"used first column '{tcol}' as the time column")
    try:
        df[tcol] = pd.to_datetime(df[tcol])
    except Exception as err:                       # noqa: BLE001 - report, do not crash
        notes.append(f"could not parse column '{tcol}' as datetimes: {err}")
    df = df.set_index(tcol)
    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        notes.append(f"{n_dup} duplicate timestamps dropped (first kept)")
        df = df[~df.index.duplicated(keep="first")]
    return df, notes


def has_time(df) -> bool:
    return isinstance(df.index, pd.DatetimeIndex)


def occupied_mask(df):
    """Occupied rows from SYS_CTL, else SF_SPD_DM, else all rows. Returns (mask, label)."""
    if "SYS_CTL" in df:
        return df["SYS_CTL"] == 1, "SYS_CTL == 1"
    if "SF_SPD_DM" in df:
        return df["SF_SPD_DM"] == 1, "SF_SPD_DM == 1 (SYS_CTL not found)"
    return pd.Series(True, index=df.index), "ALL rows (no occupancy flag found)"


def mixing_terms(df, min_delta_k):
    """Return (f, well_conditioned, dT_K) from the three temperatures, or None."""
    if not all(c in df for c in ("OA_TEMP", "RA_TEMP", "MA_TEMP")):
        return None
    d_f = df["OA_TEMP"] - df["RA_TEMP"]
    d_k = d_f * F_TO_K
    f = (df["MA_TEMP"] - df["RA_TEMP"]) / d_f.where(d_k.abs() >= 1e-3)
    return f, d_k.abs() >= min_delta_k, d_k


def is_num(s) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:.4g}")


def pct(a, q):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


# --------------------------------------------------------------------------- analyses
def column_stats(df, occ) -> pd.DataFrame:
    """Per-column facts. min/mean/max/std are over OCCUPIED rows."""
    d_occ = df[occ]
    rows = {}
    for c in df.columns:
        s = df[c]
        row = {"dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
               "nunique_all_rows": int(s.nunique(dropna=True))}
        if is_num(s) and len(d_occ):
            so = d_occ[c]
            row.update(min=so.min(), mean=so.mean(), max=so.max(), std=so.std())
        row["flag"] = "CONST" if row["nunique_all_rows"] <= 1 else ""
        rows[c] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def occupancy_lines(df, occ):
    """Describe the occupancy flags and the weekly schedule."""
    lines = []
    for flag in ("SYS_CTL", "SF_SPD_DM"):
        if flag in df:
            vc = df[flag].value_counts(dropna=False).to_dict()
            lines.append(f"{flag} value counts: {vc}")
    if "SYS_CTL" in df and "SF_SPD_DM" in df:
        disagree = float((df["SYS_CTL"] != df["SF_SPD_DM"]).mean())
        lines.append(f"SYS_CTL and SF_SPD_DM disagree on {100 * disagree:.2f}% of rows")
    if has_time(df):
        dow, hr = df.index.dayofweek, df.index.hour
        for name, sel in (("Mon-Fri", dow < 5), ("Sat", dow == 5), ("Sun", dow == 6)):
            if not sel.any():
                continue
            share = occ[sel].groupby(hr[sel]).mean()
            on = share[share > 0.5].index
            lines.append(f"{name}: occupied hours (share > 0.5): "
                         + (f"{on.min()}:00 to {on.max() + 1}:00" if len(on) else "never"))
    lines.append(f"Occupied rows by rule used here: {int(occ.sum())} of {len(df)} "
                 f"({100 * occ.mean():.1f}%)")
    return lines


def fraction_summary(df, occ, min_delta_k):
    """How the mixing-equation fraction behaves on occupied rows."""
    terms = mixing_terms(df, min_delta_k)
    if terms is None:
        return None
    f, wc, d_k = terms
    o = occ & f.notna()
    wco = occ & wc
    fw = f[wco]
    plaus = (fw >= -PLAUSIBLE_TOL) & (fw <= 1 + PLAUSIBLE_TOL)
    return {
        "n_occupied": int(occ.sum()),
        "share_well_conditioned": float(wco.sum() / max(occ.sum(), 1)),
        "dT_K_p10": pct(d_k[occ].abs(), 10), "dT_K_p50": pct(d_k[occ].abs(), 50),
        "dT_K_p90": pct(d_k[occ].abs(), 90),
        "f_wc_p05": pct(fw, 5), "f_wc_median": pct(fw, 50), "f_wc_p95": pct(fw, 95),
        "share_f_plausible_wc": float(plaus.mean()) if len(fw) else float("nan"),
        "n_finite_f_occupied": int(o.sum()),
    }


def monthly_table(df, occ, min_delta_k):
    """Occupied rows, conditioning share, outdoor temperature and fraction by month."""
    terms = mixing_terms(df, min_delta_k)
    if terms is None or not has_time(df):
        return None
    f, wc, _ = terms
    g = pd.DataFrame({"occ": occ, "wc": wc, "oa_degC": (df["OA_TEMP"] - 32.0) * 5 / 9,
                      "f_wc": f.where(wc)})
    g = g[g["occ"]]
    by = g.groupby(g.index.month)
    return pd.DataFrame({
        "n_occupied": by.size(),
        "share_well_conditioned": by["wc"].mean(),
        "mean_OA_degC": by["oa_degC"].mean(),
        "median_f_well_cond": by["f_wc"].median(),
    })


def discover_correlations(df, occ, min_delta_k, top=15):
    """Rank logged signals by Spearman correlation with the mixing fraction.

    This is a search for a usable ground truth: a flow or damper signal that
    tracks the outdoor air fraction. Temperatures appear near the top by
    construction (they define the fraction), so look past them.
    """
    terms = mixing_terms(df, min_delta_k)
    if terms is None:
        return None
    f, wc, _ = terms
    sel = occ & wc & f.notna() & (f.abs() < 5)
    sub = df.loc[sel]
    if len(sub) < 100:
        return None
    sub = sub.iloc[:: max(1, len(sub) // 100_000)]      # cap size for speed
    cand = sub.select_dtypes("number").copy()
    if "SA_CFM" in cand and "RA_CFM" in cand:
        sa = cand["SA_CFM"].replace(0, np.nan)
        cand["DERIVED (SA_CFM-RA_CFM)/SA_CFM"] = (cand["SA_CFM"] - cand["RA_CFM"]) / sa
        cand["DERIVED RA_CFM/SA_CFM"] = cand["RA_CFM"] / sa
    cand = cand.replace([np.inf, -np.inf], np.nan)
    cand = cand.loc[:, cand.std() > 0]
    rho = cand.corrwith(f.loc[sub.index], method="spearman").dropna()
    return rho.reindex(rho.abs().sort_values(ascending=False).index).head(top)


def compare_to_reference(ref, other, tol=1e-6):
    """Per-column difference between a bias file and the fault-free file."""
    same = ref.index.equals(other.index)
    common = ref.index if same else ref.index.intersection(other.index)
    rows = []
    for c in ref.columns:
        if c not in other.columns or not (is_num(ref[c]) and is_num(other[c])):
            continue
        a = ref[c] if same else ref[c].reindex(common)
        b = other[c] if same else other[c].reindex(common)
        d = b.to_numpy(float) - a.to_numpy(float)
        ad = np.abs(d)
        if not np.isfinite(ad).any():
            continue
        rows.append({"column": c, "share_rows_differ": float(np.mean(ad > tol)),
                     "max_abs_diff": float(np.nanmax(ad)), "mean_diff": float(np.nanmean(d))})
    table = pd.DataFrame(rows)
    return table, int(len(common))


def oa_temp_offset(ref, other):
    """Is the difference in OA_TEMP between files a constant? Returns lines."""
    if "OA_TEMP" not in ref or "OA_TEMP" not in other:
        return ["OA_TEMP missing in one of the files"]
    same = ref.index.equals(other.index)
    common = ref.index if same else ref.index.intersection(other.index)
    a = ref["OA_TEMP"] if same else ref["OA_TEMP"].reindex(common)
    b = other["OA_TEMP"] if same else other["OA_TEMP"].reindex(common)
    d_k = ((b - a) * F_TO_K).dropna()
    if d_k.empty:
        return ["no overlapping OA_TEMP values"]
    top = d_k.round(2).value_counts().head(4)
    return [f"OA_TEMP offset vs fault-free (K): mean {d_k.mean():.4g}, std {d_k.std():.3g}, "
            f"min {d_k.min():.4g}, max {d_k.max():.4g}",
            "most common offsets (K): " + ", ".join(f"{k:g} ({100 * v / len(d_k):.1f}%)"
                                                     for k, v in top.items())]


def regression_design(df, feats):
    """y = T_mixed - T_return (K); X columns: dT, feat*dT for each feature."""
    d_k = (df["OA_TEMP"] - df["RA_TEMP"]) * F_TO_K
    y = ((df["MA_TEMP"] - df["RA_TEMP"]) * F_TO_K).to_numpy(float)
    cols = [d_k.to_numpy(float)] + [(df[c] * d_k).to_numpy(float) for c in feats]
    return y, np.column_stack(cols)


def fit_reference(ref, occ, min_delta_k):
    """Fit the linearised model on the FIRST half of occupied well-conditioned rows."""
    terms = mixing_terms(ref, min_delta_k)
    feats = [c for c in FEATURES if c in ref]
    if terms is None or not has_time(ref):
        return None
    _, wc, _ = terms
    rows = ref.index[occ & wc]
    if len(rows) < 100:
        return None
    t_split = rows[len(rows) // 2]
    train = ref[occ & wc & (ref.index <= t_split)]
    y, X = regression_design(train, feats)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return {"coef": coef, "feats": feats, "t_split": t_split}


def residual_preview(model, df, occ, min_delta_k, bias_k):
    """Residual of the fitted model on the SECOND half of one file."""
    terms = mixing_terms(df, min_delta_k)
    if model is None or terms is None:
        return None
    _, wc, _ = terms
    test = df[occ & wc & (df.index > model["t_split"])]
    if len(test) < 10:
        return None
    y, X = regression_design(test, model["feats"])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    resid = y - X @ model["coef"]
    # Model's fitted outdoor-air fraction per row: a + b*damper + c*fan.
    f_fit = model["coef"][0] + sum(model["coef"][i + 1] * test[c].to_numpy(float)[ok]
                                   for i, c in enumerate(model["feats"]))
    return {"n_eval": int(ok.sum()), "mean_resid_K": float(resid.mean()),
            "median_resid_K": float(np.median(resid)), "std_resid_K": float(resid.std()),
            "expected_shift_K": float(-bias_k * np.nanmean(f_fit)) if bias_k is not None
            else float("nan")}


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--min-delta", type=float, default=config.MIN_DELTA_K)
    args = parser.parse_args(argv)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    order = ["fault_free"] + [c for c in config.FILES if c != "fault_free"]
    files, per_case, notes_all = {}, {}, []
    ff = ff_occ = model = None
    ff_stats = occ_lines = monthly = corr = None
    diffs, offsets, residuals, flags = {}, {}, {}, []

    for case in order:
        if case not in config.FILES:
            continue
        path = args.data_dir / config.FILES[case]
        if not path.exists():
            notes_all.append(f"[skip] {case}: {path} not found")
            continue
        df, notes = load_raw(path)
        notes_all += [f"{case}: {n}" for n in notes]
        occ, occ_label = occupied_mask(df)
        files[case] = {
            "rows": len(df), "columns": len(df.columns),
            "start": str(df.index.min()) if has_time(df) else "n/a",
            "end": str(df.index.max()) if has_time(df) else "n/a",
            "median_step": (str(pd.Series(df.index[1:] - df.index[:-1]).median())
                            if has_time(df) and len(df) > 1 else "n/a"),
            "sorted": bool(df.index.is_monotonic_increasing),
            "occupancy_rule": occ_label,
        }
        per_case[case] = fraction_summary(df, occ, args.min_delta)

        if case == "fault_free":
            ff, ff_occ = df, occ
            ff_stats = column_stats(df, occ)
            ff_stats.to_csv(config.TABLES_DIR / "column_stats_fault_free.csv")
            occ_lines = occupancy_lines(df, occ)
            monthly = monthly_table(df, occ, args.min_delta)
            corr = discover_correlations(df, occ, args.min_delta)
            model = fit_reference(df, occ, args.min_delta)
        elif ff is not None:
            table, n_common = compare_to_reference(ff, df)
            changed = table[table["share_rows_differ"] > 0].sort_values(
                "share_rows_differ", ascending=False) if len(table) else table
            diffs[case] = (changed, n_common, len(table))
            offsets[case] = oa_temp_offset(ff, df)
            if set(ff.columns) != set(df.columns):
                flags.append(f"{case}: columns differ from fault-free: "
                             f"only-fault-free {sorted(set(ff.columns) - set(df.columns))}, "
                             f"only-this {sorted(set(df.columns) - set(ff.columns))}")
        residuals[case] = residual_preview(model, df, occ, args.min_delta,
                                           config.BIAS_K.get(case)) if model else None
        if case != "fault_free":
            del df                                  # keep memory low: only fault-free is kept
        print(f"[read] {case}: {files[case]['rows']} rows")

    if not files:
        print("No files found in", args.data_dir)
        return 1

    # ----- automatic flags
    if ff_stats is not None:
        const = ff_stats.index[ff_stats["flag"] == "CONST"].tolist()
        if const:
            flags.insert(0, "CONSTANT columns in fault-free file (cannot be used as signals "
                            f"or ground truth): {const}")
        for c in ("OA_CFM", "SA_CFM", "RA_CFM"):
            if c not in ff_stats.index:
                flags.append(f"{c} not in the file")
    if len({v["rows"] for v in files.values()}) > 1:
        flags.append("files have different row counts: " +
                     ", ".join(f"{k}={v['rows']}" for k, v in files.items()))
    for case, info in files.items():
        if not info["sorted"]:
            flags.append(f"{case}: timestamps are not sorted")
    if ff is not None and "OA_DMPR_DM" not in ff:
        flags.append("OA_DMPR_DM not found: detectability preview uses fewer features")

    # ----- build report
    L = []
    def out(t=""):
        L.append(t)
    def head(title):
        out(); out("=" * 78); out(title); out("=" * 78)

    out("AHU DATA REPORT (raw column names; temperatures degF, flows CFM)")
    out(f"data dir: {args.data_dir}    min_delta: {args.min_delta:g} K    "
        f"time column: {config.TIME_COL}")
    head("0. FLAGS")
    for t in (flags or ["(none)"]) + notes_all:
        out(f"- {t}")

    head("1. FILES")
    out(fmt(pd.DataFrame(files).T))

    if ff_stats is not None:
        head("2. EVERY COLUMN, fault-free file (min/mean/max/std over OCCUPIED rows)")
        out(fmt(ff_stats))
        head("3. OCCUPANCY (fault-free)")
        for t in occ_lines:
            out(t)

    head("4. MIXING-EQUATION FRACTION f = (T_mixed - T_return)/(T_outdoor - T_return)")
    out("Occupied rows. well-conditioned = |T_outdoor - T_return| >= min_delta. "
        "In bias files T_outdoor is the biased reading.")
    ok_cases = {k: v for k, v in per_case.items() if v}
    if ok_cases:
        out(fmt(pd.DataFrame(ok_cases).T))
    else:
        out("Cannot compute: OA_TEMP, RA_TEMP or MA_TEMP missing.")
    if monthly is not None:
        out(); out("By month (fault-free, occupied rows):"); out(fmt(monthly))
    if corr is not None:
        out()
        out("Columns most correlated (Spearman) with f, fault-free, occupied well-conditioned rows."
            " Temperatures are expected here; look for FLOW or DAMPER signals that could"
            " serve as a ground truth:")
        out(fmt(corr.to_frame("spearman_rho")))

    if diffs:
        head("5. BIAS FILES vs FAULT-FREE")
        for case, (changed, n_common, n_cmp) in diffs.items():
            out(f"--- {case}: {n_common} common rows; {len(changed)} of {n_cmp} numeric "
                f"columns differ")
            for t in offsets[case]:
                out("    " + t)
            if len(changed):
                out(fmt(changed.head(12).set_index("column")))
                if len(changed) > 12:
                    out(f"    ... and {len(changed) - 12} more differing columns")

    if any(residuals.values()):
        head("6. DETECTABILITY PREVIEW")
        coef = model["coef"]
        names = ["a (dT)"] + [f"{c}*dT" for c in model["feats"]]
        out("Model: (T_mixed - T_return) = (a + b*damper + c*fan) * (T_outdoor - T_return)")
        out("Fitted on the FIRST half of fault-free occupied well-conditioned rows: "
            + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, coef)))
        out("Residual (K) on the SECOND half of each file. If a bias is detectable, "
            "mean_resid_K moves away from the fault-free value by about expected_shift_K.")
        res = pd.DataFrame({k: v for k, v in residuals.items() if v}).T
        out(fmt(res))
        base = residuals.get("fault_free")
        if base and base["std_resid_K"] > 0:
            out("shift of mean residual vs fault-free, in fault-free residual std units:")
            for k, v in residuals.items():
                if v and k != "fault_free":
                    out(f"    {k}: {(v['mean_resid_K'] - base['mean_resid_K']) / base['std_resid_K']:.3g}")

    text = "\n".join(L)
    (config.RESULTS_DIR / "data_report.txt").parent.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "data_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nReport saved to {config.RESULTS_DIR / 'data_report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
