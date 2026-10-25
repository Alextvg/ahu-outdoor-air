"""Baseline physics for outdoor air estimation and ventilation requirements.

Mixed air is a blend of outdoor and return air, so
``(T_mixed - T_return) = f * (T_outdoor - T_return)`` where ``f`` is the
outdoor air fraction. This module holds that baseline and the ASHRAE 62.1
breathing-zone calculation used as the compliance target.
"""
from __future__ import annotations

import numpy as np


def oa_fraction_from_temps(t_mixed, t_return, t_outdoor, min_delta: float = 3.0):
    """Estimate the outdoor air fraction from three air temperatures.

    ``f = (T_mixed - T_return) / (T_outdoor - T_return)``. When outdoor and
    return air are close in temperature the division is ill-conditioned (small
    sensor errors become huge flow errors), so those points return NaN.

    Args:
        t_mixed: Mixed air temperature (degC).
        t_return: Return air temperature (degC).
        t_outdoor: Outdoor air temperature (degC).
        min_delta: Smallest ``|T_outdoor - T_return|`` (K) treated as reliable.
            This is a tunable placeholder; study it in the sensitivity analysis.

    Returns:
        numpy array of fractions (unitless). Values outside 0-1 are not
        clipped, since they point to sensor error or imperfect mixing.
    """
    t_mixed = np.asarray(t_mixed, dtype=float)
    t_return = np.asarray(t_return, dtype=float)
    t_outdoor = np.asarray(t_outdoor, dtype=float)
    denom = t_outdoor - t_return
    ok = np.abs(denom) >= min_delta
    safe = np.where(ok, denom, 1.0)  # avoid dividing by ~0
    return np.where(ok, (t_mixed - t_return) / safe, np.nan)
