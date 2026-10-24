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
training rows (the same quantity the other models are scored on, the position
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

F_LO, F_HI = 0.002, 0.998          # clip before the logit
FIT_RANGE = (0.005, 0.97)          # rows with f in this range are used for the fit


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


def fit_logistic(train: pd.DataFrame, damper_col: str, fit_range=FIT_RANGE) -> LogisticEstimator:
    """Fit the two-parameter curve on well-conditioned training rows.

    Args:
        train: Dataframe with ``f_phys``, ``well_conditioned`` and ``damper_col``.
        damper_col: Column holding the true damper position.
        fit_range: ``(low, high)``; only rows with ``f_phys`` in this range are used.

    Returns:
        A :class:`LogisticEstimator`.

    Raises:
        ValueError: If fewer than 10 rows qualify or the logit of ``f`` does not
            vary (so the slope cannot be found), or the fitted slope is not positive.
    """
    d = train[train["well_conditioned"]][["f_phys", damper_col]].dropna()
    d = d[(d["f_phys"] >= fit_range[0]) & (d["f_phys"] <= fit_range[1])]
    if len(d) < 10:
        raise ValueError(f"only {len(d)} rows with f in {fit_range}; cannot fit")
    z = logit(d["f_phys"].to_numpy(float))
    x = d[damper_col].to_numpy(float)
    if np.ptp(z) == 0:
        raise ValueError("logit(f) does not vary; cannot fit a slope")
    c1, c0 = np.polyfit(z, x, 1)           # position = c0 + c1 * logit(f)
    if c1 <= 0:
        raise ValueError("position does not rise with f in the training rows; check the data")
    return LogisticEstimator(
        name="logistic_curve", kind="parametric",
        params={"a": float(-c0 / c1), "b": float(1.0 / c1), "fit_range": tuple(fit_range)},
        n_train=len(d))
