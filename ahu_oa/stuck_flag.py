"""Stuck-damper flag: does the estimated damper position agree with the command?

The damper command (``OA_DMPR_DM``) is always logged, but a stuck damper does not follow it.
The estimated position comes from the temperatures only, so it tracks where the damper
really is. A damper that is working should give an estimate close to the command, and a
stuck one should not.

The rule is deliberately simple and has two settings:

1. For each row, take ``|estimate - command|`` (damper units).
2. Take the median of that gap over a trailing time window (``window``). A median ignores
   brief disagreements, such as the damper still moving after a command change.
3. Raise the flag when that median exceeds ``threshold``.

The threshold is chosen on fault-free validation data only, as the smallest value whose
false-alarm rate stays under a target (:func:`choose_threshold`), so no stuck file is used to
tune it.

Limit: a damper stuck at position ``p`` can only be caught while the command differs from ``p``
by more than the threshold. If the controller sits at ``p`` most of the time (for example a
damper stuck at its 0.10 minimum), detection is capped by physics, not by the model.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def disagreement(estimate, command) -> pd.Series:
    """Absolute gap between the estimated and commanded position.

    Args:
        estimate: Estimated positions (Series with a time index, NaN where no estimate).
        command: Commanded positions on the same index.

    Returns:
        Series of ``|estimate - command|`` with the NaN rows dropped.
    """
    est = pd.Series(estimate)
    cmd = pd.Series(command, index=est.index)
    return (est - cmd).abs().dropna()


def rolling_flag(gap: pd.Series, threshold: float, window: str = "60min",
                 min_rows: int = 20) -> pd.DataFrame:
    """Flag rows whose trailing-window median gap exceeds ``threshold``.

    Args:
        gap: Output of :func:`disagreement`, with a sorted ``DatetimeIndex``.
        threshold: Gap (damper units) above which the median counts as disagreement.
        window: Trailing window length, a pandas offset string.
        min_rows: Fewest rows in the window for a decision; otherwise the row is undecided.

    Returns:
        Dataframe indexed like ``gap`` with ``rolling_median``, ``decided`` (enough rows) and
        ``flag`` (decided and above the threshold).

    Raises:
        ValueError: If ``threshold`` is negative or ``min_rows`` is below 1.
        TypeError: If ``gap`` has no ``DatetimeIndex``.
    """
    if threshold < 0:
        raise ValueError("threshold must not be negative")
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1")
    if not isinstance(gap.index, pd.DatetimeIndex):
        raise TypeError("rolling_flag needs a DatetimeIndex")
    med = gap.rolling(window, min_periods=min_rows).median()
    decided = med.notna()
    return pd.DataFrame({"rolling_median": med, "decided": decided,
                         "flag": decided & (med > threshold)})


def flag_rates(flags: pd.DataFrame) -> dict:
    """Summarise a flag table at row and day level.

    Row level is the share of decided rows that are flagged. Day level is the share of days
    (with at least one decided row) that have at least one flagged row, which is how an
    engineer would see alarms.

    Returns:
        Dict with ``n_rows`` (all rows), ``n_decided``, ``row_flag_rate``, ``n_days`` and
        ``day_flag_rate`` (NaN where nothing was decided).
    """
    d = flags[flags["decided"]]
    out = {"n_rows": int(len(flags)), "n_decided": int(len(d)),
           "row_flag_rate": float(d["flag"].mean()) if len(d) else np.nan}
    if len(d):
        per_day = d["flag"].groupby(d.index.normalize()).any()
        out.update(n_days=int(len(per_day)), day_flag_rate=float(per_day.mean()))
    else:
        out.update(n_days=0, day_flag_rate=np.nan)
    return out


def choose_threshold(gap: pd.Series, candidates, window: str = "60min", min_rows: int = 20,
                     max_false_alarm: float = 0.01) -> float:
    """Smallest candidate threshold whose row false-alarm rate is within the target.

    Use fault-free validation data only. If no candidate meets the target, the largest one is
    returned with a warning.

    Args:
        gap: Disagreement on fault-free validation rows.
        candidates: Thresholds to try (damper units).
        window: As in :func:`rolling_flag`.
        min_rows: As in :func:`rolling_flag`.
        max_false_alarm: Highest acceptable share of fault-free rows flagged (0.01 is 1%).

    Returns:
        The chosen threshold.

    Raises:
        ValueError: If there are no candidates or ``max_false_alarm`` is outside 0 to 1.
    """
    cands = sorted(float(c) for c in candidates)
    if not cands:
        raise ValueError("no candidate thresholds")
    if not 0 <= max_false_alarm <= 1:
        raise ValueError("max_false_alarm must be between 0 and 1")
    for c in cands:
        rate = flag_rates(rolling_flag(gap, c, window, min_rows))["row_flag_rate"]
        if np.isfinite(rate) and rate <= max_false_alarm:
            return c
    warnings.warn(f"no threshold up to {cands[-1]} keeps false alarms under "
                  f"{max_false_alarm:.1%}; using the largest", stacklevel=2)
    return cands[-1]
