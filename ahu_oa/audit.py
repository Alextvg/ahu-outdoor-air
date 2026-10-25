"""Audit AHU CSV files for the damper-stuck research design.

Design being checked: learn how the damper relates to the outdoor air fraction
(from the mixing equation) on fault-free data, invert it to estimate damper
position, then test on files where the damper is stuck at a known position.
The outdoor flow column (OA_CFM) is not used.

The function returns tables (not printed text) so the calling script decides
what to save under ``results/``. Thresholds are choices, listed in ``THRESH``.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ahu_oa import loading, physics, regression, settling

THRESH = dict(
    min_delta=3.0,        # K, smallest |T_outdoor - T_return| trusted
    settle="10min",       # fan-on time before mixed air is trusted
    gap_factor=1.5,       # an interval > this x the median interval is a gap
    max_gap_share=0.05,
    stuck_tol=0.01,       # damper within this of its mode counts as stuck
    stuck_share=0.95,     # share of rows that must sit at the stuck value
    name_tol=0.02,        # tolerance between documented position and data
    min_ff_std=0.05,      # fault-free damper must move at least this much
    cov_half_width=0.05,  # window around the stuck position
    min_cov_rows=200,     # fault-free rows needed inside that window
    min_r2=0.5,           # fault-free fit of fraction on damper
    min_wc_rows=500,
    min_wc_share=0.2,
    pass_dev=0.10,        # |implied - known| damper position
    warn_dev=0.20,
    plaus_lo=-0.05, plaus_hi=1.05, min_plausible=0.95,
)
TEMPS = ["OA_TEMP_C", "RA_TEMP_C", "MA_TEMP_C"]
DAMPERS = ["OA_DMPR", "OA_DMPR_DM"]
FLOWS = ["OA_Ls", "SA_Ls"]
COMPARE = ["OA_TEMP_C", "RA_TEMP_C", "MA_TEMP_C", "OA_DMPR", "OA_DMPR_DM", "SF_SPD_DM"]


@dataclass
class AuditResult:
    """Output of :func:`audit`. All tables are dataframes.

    Attributes:
        files: One row per file: checksum, rows, duplicate relationships.
        structure: One row per analysed case: period, sampling, fan, flows,
            damper statistics and usable row counts.
        monthly: Well-conditioned usable rows per case and month.
        fits: Fault-free fraction-versus-damper fit, per damper column.
        stuck: One row per stuck case: stuck state, coverage, implied position.
        checks: PASS / WARN / FAIL / INFO checklist.
        verdict: One-line summary of the checklist.
        prepared: Case -> analysis dataframe (fan on, settled), for plotting.
        settings: Thresholds used (``THRESH`` plus overrides).
    """

    files: pd.DataFrame
    structure: pd.DataFrame
    monthly: pd.DataFrame
    fits: pd.DataFrame
    stuck: pd.DataFrame
    checks: pd.DataFrame
    verdict: str
    prepared: dict
    settings: dict


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _load(path, time_col):
    """Load in SI units with a sorted, de-duplicated index; return (df, n_dup)."""
    df = loading.load_ahu_csv(str(path), time_col=time_col)
    n_dup = int(df.index.duplicated().sum())
    return df[~df.index.duplicated()].sort_index(), n_dup


def _prep(df, th):
    """Fan-on, settled rows with delta, well_conditioned and f_phys columns."""
    on = (df["SF_SPD_DM"] == 1) if "SF_SPD_DM" in df else pd.Series(True, index=df.index)
    keep = settling.settled_mask(df.index, on, th["settle"])
    a = df[keep].dropna(subset=TEMPS).copy()
    a["delta"] = a["OA_TEMP_C"] - a["RA_TEMP_C"]
    a["well_conditioned"] = a["delta"].abs() >= th["min_delta"]
    a["f_phys"] = physics.oa_fraction_from_temps(
        a["MA_TEMP_C"], a["RA_TEMP_C"], a["OA_TEMP_C"], min_delta=th["min_delta"])
    return a, float(on.mean())


def _mode_share(s, tol):
    s = s.dropna()
    if s.empty:
        return float("nan"), float("nan")
    mode = float(s.round(2).mode().iloc[0])
    return mode, float(((s - mode).abs() <= tol).mean())


def position_from_name(path):
    """Read a stuck position from a name like ``damper_stuck_100_...`` (-> 1.0)."""
    m = re.search(r"stuck_(\d+(?:\.\d+)?)", os.path.basename(str(path)))
    if not m:
        return None
    v = float(m.group(1))
    return v / 100 if v > 1 else v


def _r2(y, yhat):
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def audit(fault_free, stuck, extra=None, time_col="Datetime", **overrides) -> AuditResult:
    """Audit the files for the damper-stuck design.

    Args:
        fault_free: Path to the fault-free CSV.
        stuck: Mapping ``label -> (path, position)``. ``position`` is the known
            stuck damper position (0-1), or None to read it from the file name.
        extra: Mapping ``label -> path`` of other files (for example the
            ``oa_bias_*`` files), checked only for duplication.
        time_col: Name of the time column.
        **overrides: Replacements for entries in ``THRESH``.

    Returns:
        An :class:`AuditResult`.
    """
    th = {**THRESH, **overrides}
    extra = extra or {}
    paths = {"fault_free": fault_free, **{k: v[0] for k, v in stuck.items()}, **extra}
    data, hashes, ndups = {}, {}, {}
    for k, p in paths.items():
        data[k], ndups[k] = _load(p, time_col)
        hashes[k] = _hash(p)
    ff_df = data["fault_free"]
    checks = []

    def check(status, name, detail):
        checks.append({"status": status, "check": name, "detail": detail})

    # ---- files and duplicates
    groups = {}
    for k, h in hashes.items():
        groups.setdefault(h, []).append(k)
    rows = []
    for k, p in paths.items():
        same = bool(k != "fault_free" and data[k].shape == ff_df.shape
                    and data[k].select_dtypes("number").equals(ff_df.select_dtypes("number")))
        rows.append({"case": k, "file": os.path.basename(str(p)), "sha256_12": hashes[k],
                     "n_rows": len(data[k]),
                     "same_bytes_as": ", ".join(x for x in groups[hashes[k]] if x != k),
                     "same_numeric_data_as_fault_free": same})
    files = pd.DataFrame(rows)
    bad = [k for k in stuck if hashes[k] == hashes["fault_free"]
           or files.loc[files["case"] == k, "same_numeric_data_as_fault_free"].iloc[0]]
    check("FAIL" if bad else "PASS", "stuck files differ from fault-free",
          f"identical to fault-free: {bad}" if bad else "all stuck files contain different data")
    dup_groups = [v for v in groups.values() if len(v) > 1]
    if dup_groups:
        check("INFO", "duplicate files", str(dup_groups))

    # ---- structure per analysed case
    prepared, struct_rows, monthly_rows = {}, [], []
    for k in ["fault_free"] + list(stuck):
        df = data[k]
        a, fan_share = _prep(df, th)
        prepared[k] = a
        dt = df.index.to_series().diff().dropna()
        med = dt.median() if len(dt) else pd.NaT
        gaps = int((dt > th["gap_factor"] * med).sum()) if len(dt) else 0
        gap_share = gaps / max(len(dt), 1)
        wc = a["well_conditioned"]
        r = {"case": k, "n_rows": len(df), "time_start": str(df.index.min()),
             "time_end": str(df.index.max()), "median_step": str(med), "n_gaps": gaps,
             "gap_share": gap_share, "n_duplicate_timestamps": ndups[k],
             "fan_on_share": fan_share, "n_usable": len(a),
             "n_well_conditioned": int(wc.sum()),
             "share_well_conditioned": float(wc.mean()) if len(a) else float("nan")}
        for c in FLOWS:
            if c in df:
                r[f"{c}_n_unique"] = int(df[c].nunique())
        for c in DAMPERS:
            if c in df:
                s = df[c].dropna()
                r.update({f"{c}_min": float(s.min()), f"{c}_max": float(s.max()),
                          f"{c}_std": float(s.std()), f"{c}_n_unique": int(s.nunique())})
        struct_rows.append(r)
        for m, n in a[wc].groupby(a[wc].index.month).size().items():
            monthly_rows.append({"case": k, "month": int(m), "n_well_conditioned": int(n)})
        check("PASS" if gap_share <= th["max_gap_share"] else "WARN", f"{k}: regular sampling",
              f"{gaps} gaps, {gap_share:.1%} of intervals")
        miss = [c for c in TEMPS if c not in df] + ([] if any(c in df for c in DAMPERS) else ["damper column"])
        check("FAIL" if miss else "PASS", f"{k}: required columns", f"missing {miss}" if miss else "present")
        if "OA_Ls_n_unique" in r and r["OA_Ls_n_unique"] <= 1:
            check("INFO", f"{k}: outdoor flow constant", "OA flow cannot be ground truth; design does not use it")
    structure, monthly = pd.DataFrame(struct_rows), pd.DataFrame(monthly_rows)

    # ---- fault-free fraction versus damper
    a_ff = prepared["fault_free"]
    wc_ff = a_ff[a_ff["well_conditioned"]]
    fit_rows, models = [], {}
    for c in [c for c in DAMPERS if c in a_ff]:
        try:
            m = regression.fit(a_ff, damper_col=c, well_conditioned_only=True)
        except (ValueError, KeyError):
            continue
        pred = m.predict_fraction(wc_ff)
        r2 = _r2(wc_ff["f_phys"].to_numpy(float), pred)
        sd = float(np.nanstd(wc_ff["f_phys"].to_numpy(float) - pred))
        models[c] = (m, r2)
        fit_rows.append({"damper_col": c, "n": m.n_train, "a": m.coefs["a"], "b": m.coefs["b"],
                         "r_squared": r2, "resid_sd_fraction": sd,
                         "approx_damper_error": sd / m.coefs["b"] if m.coefs["b"] > 0 else float("nan"),
                         "cond": m.cond})
    fits = pd.DataFrame(fit_rows)
    if not models:
        check("FAIL", "fault-free fit", "no damper column could be fitted")

    # ---- stuck files
    stuck_rows = []
    for k, (path, pos_doc) in stuck.items():
        df, a = data[k], prepared[k]
        wc = a[a["well_conditioned"]]
        p_name = position_from_name(path)
        known, source = (pos_doc, "config") if pos_doc is not None else (p_name, "filename")
        modes = {c: _mode_share(df[c], th["stuck_tol"]) for c in DAMPERS if c in df}
        row = {"case": k, "position_known": known, "position_source": source,
               "position_in_filename": p_name}
        for c, (mode, share) in modes.items():
            row[f"{c}_mode"], row[f"{c}_mode_share"] = mode, share
        stuck_cols = [c for c, (_, s) in modes.items() if s >= th["stuck_share"]]
        if not stuck_cols:
            check("FAIL", f"{k}: stuck state visible", "no damper column is constant: true state not recorded")
            stuck_rows.append(row)
            continue
        scol = stuck_cols[0]
        p_data = modes[scol][0]
        row.update({"stuck_col": scol, "position_in_data": p_data})
        p = known if known is not None else p_data
        if known is not None:
            ok = abs(known - p_data) <= th["name_tol"]
            check("PASS" if ok else "WARN", f"{k}: documented position matches data",
                  f"documented {known} ({source}), data {p_data:.2f}")
        check("PASS", f"{k}: stuck column", f"{scol} constant at {p_data:.2f}")

        tcol = scol if scol in models else (max(models, key=lambda c: models[c][1]) if models else None)
        if tcol is None:
            stuck_rows.append(row)
            continue
        near = int(((wc_ff[tcol] - p).abs() <= th["cov_half_width"]).sum())
        std_ff = float(wc_ff[tcol].std())
        row.update({"fit_col": tcol, "n_fault_free_near_position": near, "fault_free_damper_std": std_ff})
        check("PASS" if std_ff >= th["min_ff_std"] else "FAIL", "fault-free damper varies", f"std {std_ff:.3f}")
        check("PASS" if near >= th["min_cov_rows"] else "WARN", f"{k}: training covers stuck position",
              f"{near} fault-free well-conditioned rows within +-{th['cov_half_width']} of {p}")
        m, r2 = models[tcol]
        check("PASS" if (r2 >= th["min_r2"] and m.coefs["b"] > 0) else "FAIL",
              "fault-free: damper explains fraction", f"{tcol}: R2={r2:.2f}, b={m.coefs['b']:.3f}")

        share = len(wc) / max(len(a), 1)
        row.update({"n_well_conditioned": len(wc), "share_well_conditioned": share})
        check("PASS" if (len(wc) >= th["min_wc_rows"] and share >= th["min_wc_share"]) else "WARN",
              f"{k}: enough well-conditioned rows", f"{len(wc)} rows, {share:.1%} of usable")
        if wc.empty:
            stuck_rows.append(row)
            continue
        f = wc["f_phys"].to_numpy(float)
        plaus = float(((f >= th["plaus_lo"]) & (f <= th["plaus_hi"])).mean())
        check("PASS" if plaus >= th["min_plausible"] else "WARN", f"{k}: fractions plausible",
              f"{plaus:.1%} within [{th['plaus_lo']}, {th['plaus_hi']}]")
        d_hat = (f - m.coefs["a"]) / m.coefs["b"] if m.coefs["b"] > 0 else np.full_like(f, np.nan)
        q1, med, q3 = np.nanpercentile(d_hat, [25, 50, 75])
        dev = abs(med - p)
        row.update({"fraction_mean": float(np.nanmean(f)), "fraction_sd": float(np.nanstd(f)),
                    "fraction_expected_from_fault_free": m.coefs["a"] + m.coefs["b"] * p,
                    "implied_median": float(med), "implied_q1": float(q1), "implied_q3": float(q3),
                    "implied_share_within_tol": float((np.abs(d_hat - p) <= th["pass_dev"]).mean()),
                    "implied_deviation": float(dev), "fraction_plausible_share": plaus})
        check("PASS" if dev <= th["pass_dev"] else ("WARN" if dev <= th["warn_dev"] else "FAIL"),
              f"{k}: implied position matches known", f"median {med:.3f} vs {p} (dev {dev:.3f})")

        both = a.index.intersection(a_ff.index)
        row["n_shared_timestamps"] = int(len(both))
        if len(both):
            ma = a.loc[both, "MA_TEMP_C"] - a_ff.loc[both, "MA_TEMP_C"]
            same_w = {c: float(((df.loc[both, c] - ff_df.loc[both, c]).abs() < 1e-6).mean())
                      for c in COMPARE if c in df and c in ff_df}
            row.update({"ma_temp_shift_mean_K": float(ma.mean()), "ma_temp_shift_sd_K": float(ma.std()),
                        **{f"identical_share_{c}": v for c, v in same_w.items()}})
            w_ok = same_w.get("OA_TEMP_C", 0) > 0.99 and same_w.get("RA_TEMP_C", 0) > 0.99
            check("INFO", f"{k}: weather matches fault-free",
                  "OA/RA temperatures identical at shared times (paired comparison possible)" if w_ok
                  else "weather differs from fault-free, so differences mix fault and weather")
        else:
            check("INFO", f"{k}: weather matches fault-free", "no shared timestamps")
        stuck_rows.append(row)

    checks_df = pd.DataFrame(checks)
    fails = int((checks_df["status"] == "FAIL").sum())
    warns = int((checks_df["status"] == "WARN").sum())
    if fails:
        verdict = f"NOT FIT as designed: {fails} FAIL, {warns} WARN"
    elif warns:
        verdict = f"FIT WITH CAVEATS: {warns} WARN"
    else:
        verdict = "FIT: all checks pass"
    return AuditResult(files, structure, monthly, fits, pd.DataFrame(stuck_rows),
                       checks_df, verdict, prepared, th)


def _n(v, nd=3):
    """Format a number, or '-' when missing."""
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}g}"


def format_report(r: AuditResult) -> str:
    """Return a short plain-text report with only what decides the question.

    Designed to be pasted or sent as one file: the verdict first, then one
    block per case, then every check that is not a plain PASS.
    """
    s = r.settings
    L = [f"DAMPER AUDIT  (min_delta {s['min_delta']} K, settle {s['settle']})",
         f"VERDICT: {r.verdict}", "", "FILES"]
    for _, f in r.files.iterrows():
        L.append(f"  {f['case']}: {f['file']}, {f['n_rows']} rows")
    for _, g in r.files.groupby("sha256_12"):
        if len(g) > 1:
            L.append("  IDENTICAL FILES: " + " = ".join(g["case"]))
    for _, f in r.files[r.files["same_numeric_data_as_fault_free"]].iterrows():
        L.append(f"  same data as fault-free: {f['case']}")

    L += ["", "DATA"]
    for _, c in r.structure.iterrows():
        L.append(f"  {c['case']}: {c['time_start'][:10]} to {c['time_end'][:10]}, step {str(c['median_step']).replace('0 days ', '')}, "
                 f"gaps {c['n_gaps']}, fan on {_n(100 * c['fan_on_share'], 3)}%")
        L.append(f"    usable {c['n_usable']}, well-conditioned {c['n_well_conditioned']} "
                 f"({_n(100 * c['share_well_conditioned'], 3)}%)")
        m = r.monthly[r.monthly["case"] == c["case"]]
        L.append("    by month: " + " ".join(f"{int(a)}:{int(b)}" for a, b in
                                              zip(m["month"], m["n_well_conditioned"])))
        const = [k for k in ("OA_Ls", "SA_Ls") if c.get(f"{k}_n_unique", 2) <= 1]
        if const:
            L.append(f"    constant columns: {', '.join(const)}")
        for d in DAMPERS:
            if f"{d}_min" in c:
                L.append(f"    {d}: {_n(c[f'{d}_min'])} to {_n(c[f'{d}_max'])}, std {_n(c[f'{d}_std'])}")

    L += ["", "FAULT-FREE: fraction = a + b * damper (well-conditioned rows)"]
    for _, f in r.fits.iterrows():
        L.append(f"  {f['damper_col']}: a {_n(f['a'])}, b {_n(f['b'])}, R2 {_n(f['r_squared'])}, "
                 f"damper error ~{_n(f['approx_damper_error'])}, cond {_n(f['cond'])}")

    for _, k in r.stuck.iterrows():
        L += ["", f"STUCK: {k['case']}  (known position {_n(k['position_known'])}, from {k['position_source']})"]
        if "stuck_col" not in k or pd.isna(k.get("stuck_col")):
            L.append("  no damper column is constant: stuck state not recorded")
            continue
        L.append(f"  constant column {k['stuck_col']} at {_n(k['position_in_data'])}")
        L.append(f"  fault-free rows near that position: {k.get('n_fault_free_near_position', '-')}")
        if pd.notna(k.get("implied_median")):
            L.append(f"  fraction mean {_n(k['fraction_mean'])} (sd {_n(k['fraction_sd'])}), "
                     f"expected {_n(k['fraction_expected_from_fault_free'])}")
            L.append(f"  implied position median {_n(k['implied_median'])} "
                     f"[IQR {_n(k['implied_q1'])}, {_n(k['implied_q3'])}], "
                     f"{_n(100 * k['implied_share_within_tol'], 3)}% within +-{s['pass_dev']}")
        if pd.notna(k.get("ma_temp_shift_mean_K")):
            L.append(f"  vs fault-free at {k['n_shared_timestamps']} shared times: mixed-air shift "
                     f"{_n(k['ma_temp_shift_mean_K'])} K, OA temp identical "
                     f"{_n(100 * k.get('identical_share_OA_TEMP_C', float('nan')), 3)}%")

    c = r.checks
    L += ["", f"CHECKS: {int((c['status'] == 'PASS').sum())} PASS"]
    for _, x in c[c["status"].isin(["FAIL", "WARN"])].iterrows():
        L.append(f"  [{x['status']}] {x['check']}: {x['detail']}")
    for _, x in c[c["status"] == "INFO"].iterrows():
        L.append(f"  [INFO] {x['check']}: {x['detail']}")
    return "\n".join(L) + "\n"
