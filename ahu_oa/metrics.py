"""Error metrics and decision rates for scoring outdoor airflow estimates.

All functions are pure: they take arrays and return numbers or dictionaries.
Rows where either the truth or the estimate is NaN (for example ill-conditioned
points the baseline refuses to estimate) are left out of the error statistics,
and the share that was scored is reported as ``coverage``.
"""
from __future__ import annotations

import numpy as np


def error_summary(y_true, y_pred) -> dict:
    """Summarise the error of an estimate against ground truth.

    Error is defined as ``y_pred - y_true``, so a positive bias means the
    estimate is too high.

    Args:
        y_true: Ground truth values (any unit; results are in the same unit).
        y_pred: Estimated values, same length as ``y_true``. NaN is allowed.

    Returns:
        Dictionary with ``n_total`` (rows given), ``n_scored`` (rows with both
        values finite), ``coverage`` (n_scored / n_total), ``mae``, ``rmse``,
        ``bias`` (mean error) and ``nmae_pct`` (MAE as a percentage of the mean
        true value, which avoids dividing by near-zero individual values).
        Statistics are NaN when nothing could be scored.

    Raises:
        ValueError: If the inputs differ in length.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    n_total, n_scored = int(y_true.size), int(ok.sum())
    out = {
        "n_total": n_total,
        "n_scored": n_scored,
        "coverage": n_scored / n_total if n_total else float("nan"),
        "mae": float("nan"), "rmse": float("nan"),
        "bias": float("nan"), "nmae_pct": float("nan"),
    }
    if n_scored == 0:
        return out
    err = y_pred[ok] - y_true[ok]
    out["mae"] = float(np.mean(np.abs(err)))
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["bias"] = float(np.mean(err))
    mean_true = float(np.mean(y_true[ok]))
    if mean_true != 0.0:
        out["nmae_pct"] = 100.0 * out["mae"] / mean_true
    return out


def summarise_by_conditioning(y_true, y_pred, well_conditioned) -> dict:
    """Score an estimate on all rows, and separately by conditioning.

    Args:
        y_true: Ground truth values.
        y_pred: Estimated values.
        well_conditioned: Boolean array, True where the outdoor/return
            temperature difference is large enough to trust the mixing equation.

    Returns:
        Dictionary with keys ``"all"``, ``"well_conditioned"`` and
        ``"ill_conditioned"``, each holding the output of :func:`error_summary`.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(well_conditioned, dtype=bool)
    return {
        "all": error_summary(y_true, y_pred),
        "well_conditioned": error_summary(y_true[mask], y_pred[mask]),
        "ill_conditioned": error_summary(y_true[~mask], y_pred[~mask]),
    }


def compliance_rates(true_flow, est_flow, required_flow) -> dict:
    """Compare the pass/fail decision made from an estimate with the true one.

    A row *passes* when its outdoor airflow is at least ``required_flow``.

    Definitions (conditional, so they are not swamped by the majority class):
      * false pass rate = P(estimate passes | truth fails)
      * false fail rate = P(estimate fails | truth passes)
    A rate is NaN when its condition never occurs (for example if the truth
    fails on every row, there is no false fail rate to measure).

    Args:
        true_flow: True outdoor airflow (L/s).
        est_flow: Estimated outdoor airflow (L/s). NaN rows are not decided.
        required_flow: Minimum required outdoor airflow (L/s), a single number.

    Returns:
        Dictionary with ``n_decisions``, ``true_pass_rate``, ``est_pass_rate``,
        ``false_pass_rate``, ``false_fail_rate`` and ``agreement`` (share of
        decisions where estimate and truth agree).
    """
    true_flow = np.asarray(true_flow, dtype=float)
    est_flow = np.asarray(est_flow, dtype=float)
    ok = np.isfinite(true_flow) & np.isfinite(est_flow)
    t_pass = true_flow[ok] >= required_flow
    e_pass = est_flow[ok] >= required_flow
    n = int(ok.sum())
    nan = float("nan")
    n_true_fail = int((~t_pass).sum())
    n_true_pass = int(t_pass.sum())
    return {
        "n_decisions": n,
        "true_pass_rate": float(t_pass.mean()) if n else nan,
        "est_pass_rate": float(e_pass.mean()) if n else nan,
        "false_pass_rate": float((e_pass & ~t_pass).sum() / n_true_fail) if n_true_fail else nan,
        "false_fail_rate": float((~e_pass & t_pass).sum() / n_true_pass) if n_true_pass else nan,
        "agreement": float((e_pass == t_pass).mean()) if n else nan,
    }
