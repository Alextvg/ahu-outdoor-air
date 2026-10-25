"""Two-parameter physical damper curve: f = 1 / (1 + exp(-(a + b * position))).

Why this shape. In a mixing box the outdoor air damper and the return air damper
work in opposition. Opposed-blade dampers have an approximately exponential
flow characteristic: their resistance grows roughly exponentially as they close.
The ratio of outdoor to return flow, ``f / (1 - f)``, is then roughly exponential
in the damper position, so its logarithm (the *logit* of ``f``) is a straight line:

    logit(f) = ln(f / (1 - f)) = a + b * position.

Check on the stuck-file means in the audit (f = 0.0156, 0.0437, 0.675 at positions
0.10, 0.25, 0.75): the logit is -4.15, -3.08, +0.73, a slope of 7.1 then 7.6 per
unit position, close to straight. That is also why a straight line in ``f`` fails:
it cannot be flat at low positions and steep in the middle.

Estimator. The curve is turned around to give the position from ``f``:

    position = (logit(f) - a) / b.

``a`` and ``b`` come from least squares of the position on ``logit(f)`` over the
training rows (``method="inverse"``; ``"forward"`` fits ``f`` itself, see ``fit_logistic``;
``select_logistic`` picks the method and fit range on the validation block) (the same quantity the other models are scored on, the position
error), then are re-expressed as ``a = -c0 / c1``, ``b = 1 / c1``. Two numbers
describe the whole model, and ``b`` has a meaning: how fast the flow ratio changes
with opening (the damper's exponential characteristic).

Limits, stated so they can be reported. ``f`` is clipped to [``f_lo``, ``f_hi``]
before the logit, because the logit is infinite at 0 and 1 and noise puts ``f``
slightly outside [0, 1]. The fit uses only rows with ``f`` inside ``fit_range``:
rows saturated at ``f`` near 1 (damper fully open, return damper shut) carry
little information about the curve and a logistic never reaches 1 exactly. The
output is limited to [0, 1].

Same interface as the other estimators (``name``, ``kind``, ``params``,
``n_train``, ``predict``) so ``scripts/run_damper_model.py`` scoring works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

F_LO, F_HI = 0.002, 0.998          # clip before the logit
FIT_RANGE = (0.005, 0.97)          # rows with f in this range are used for the fit
METHODS = ("inverse", "forward")
DEFAULT_LOWS = (0.002, 0.005, 0.01, 0.02)      # candidate lower ends of the fit range
DEFAULT_HIGHS = (0.90, 0.95, 0.97, 0.99, 1.0)  # candidate upper ends of the fit range


def logit(f) -> np.ndarray:
    """ln(f / (1 - f)) with ``f`` clipped to [F_LO, F_HI]; NaN stays NaN."""
    f = np.clip(np.asarray(f, dtype=float), F_LO, F_HI)
    return np.log(f / (1.0 - f))


@dataclass
class LogisticEstimator:
    """Fitted ``logit(f) = a + b * position``, used in the inverse direction.

    Attributes:
        name: Label used in result tables.
        kind: Always ``"parametric"``.
        params: ``{"a": intercept, "b": slope of logit(f) per unit position,
            "fit_range": (f_min, f_max)}``.
        n_train: Rows used in the fit.
    """

    name: str
    kind: str
    params: dict
    n_train: int

    def predict(self, f) -> np.ndarray:
        """Damper position implied by fractions ``f``, limited to [0, 1] (NaN stays NaN)."""
        f = np.asarray(f, dtype=float)
        out = (logit(f) - self.params["a"]) / self.params["b"]
        out = np.clip(out, 0.0, 1.0)
        return np.where(np.isfinite(f), out, np.nan)


def fit_logistic(train: pd.DataFrame, damper_col: str, fit_range=FIT_RANGE,
                 method: str = "inverse") -> LogisticEstimator:
    """Fit the two-parameter curve on well-conditioned training rows.

    Args:
        train: Dataframe with ``f_phys``, ``well_conditioned`` and ``damper_col``.
        damper_col: Column holding the true damper position.
        fit_range: ``(low, high)``; with ``method="inverse"`` only rows with ``f_phys``
            in this range are used.
        method: ``"inverse"`` regresses position on ``logit(f)`` (minimises position
            error, the quantity scored). ``"forward"`` fits ``f = sigmoid(a + b *
            position)`` by robust nonlinear least squares on all rows (soft-L1 loss,
            so the lag rows with high ``f`` at a low position count less), then
            inverts it. The forward fit is the physical model taken literally; it is
            weighted towards large ``f``, where the curve is steep.

    Returns:
        A :class:`LogisticEstimator`.

    Raises:
        ValueError: On an unknown ``method``, fewer than 10 rows, a logit of ``f``
            that does not vary, or a fitted slope that is not positive.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    d = train[train["well_conditioned"]][["f_phys", damper_col]].dropna()
    if method == "inverse":
        d = d[(d["f_phys"] >= fit_range[0]) & (d["f_phys"] <= fit_range[1])]
    if len(d) < 10:
        raise ValueError(f"only {len(d)} usable rows; cannot fit")
    x = d[damper_col].to_numpy(float)
    f = d["f_phys"].to_numpy(float)
    z = logit(f)
    if np.ptp(z) == 0:
        raise ValueError("logit(f) does not vary; cannot fit a slope")
    c1, c0 = np.polyfit(z, x, 1)           # position = c0 + c1 * logit(f)
    if c1 <= 0:
        raise ValueError("position does not rise with f in the training rows; check the data")
    a, b = -c0 / c1, 1.0 / c1
    if method == "forward":
        fc = np.clip(f, 0.0, 1.0)
        sol = least_squares(lambda p: 1.0 / (1.0 + np.exp(-(p[0] + p[1] * x))) - fc,
                            x0=[a, b], loss="soft_l1", f_scale=0.05)
        a, b = float(sol.x[0]), float(sol.x[1])
        if b <= 0:
            raise ValueError("forward fit gave a non-positive slope; check the data")
    return LogisticEstimator(
        name="logistic_curve", kind="parametric",
        params={"a": float(a), "b": float(b), "fit_range": tuple(fit_range), "method": method},
        n_train=len(d))


def select_logistic(train: pd.DataFrame, val: pd.DataFrame, damper_col: str,
                    lows=DEFAULT_LOWS, highs=DEFAULT_HIGHS, methods=METHODS):
    """Choose the fit settings (method and fit range) by validation error.

    Every combination is fitted on ``train`` and scored on the well-conditioned rows
    of ``val`` (mean absolute position error). The forward method ignores the range, so
    it is tried once. The test block and stuck files are not used.

    Returns:
        Tuple ``(best_estimator, table)``; ``table`` has one row per combination with
        ``method``, ``low``, ``high``, ``a``, ``b``, ``val_mae`` and ``chosen``.

    Raises:
        ValueError: If no combination could be fitted.
    """
    held = val[val["well_conditioned"]]
    truth = held[damper_col].to_numpy(float)
    rows, fits = [], []
    for m in methods:
        ranges = [(np.nan, np.nan)] if m == "forward" else [(lo, hi) for lo in lows for hi in highs if lo < hi]
        for lo, hi in ranges:
            try:
                est = fit_logistic(train, damper_col, fit_range=(lo, hi) if m == "inverse" else FIT_RANGE,
                                   method=m)
            except ValueError:
                continue
            err = np.abs(est.predict(held["f_phys"].to_numpy(float)) - truth)
            err = err[np.isfinite(err)]
            rows.append({"method": m, "low": lo, "high": hi, "a": est.params["a"], "b": est.params["b"],
                         "val_mae": float(err.mean()) if err.size else np.nan})
            fits.append(est)
    table = pd.DataFrame(rows)
    if table.empty or table["val_mae"].isna().all():
        raise ValueError("no logistic fit could be scored; check the data")
    best = int(table["val_mae"].idxmin())
    table["chosen"] = table.index == best
    return fits[best], table
