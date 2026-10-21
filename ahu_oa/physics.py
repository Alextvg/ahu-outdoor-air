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


def oa_flow_from_fraction(fraction, supply_flow_ls):
    """Convert an outdoor air fraction into outdoor airflow (L/s).

    Args:
        fraction: Outdoor air fraction (unitless).
        supply_flow_ls: Supply airflow (L/s).

    Returns:
        Outdoor airflow in L/s.
    """
    return np.asarray(fraction, dtype=float) * np.asarray(supply_flow_ls, dtype=float)


def default_occupancy(area_m2: float, density_per_1000ft2: float = 5.0) -> float:
    """Estimate the number of people from floor area and a default density.

    The 5 people per 1000 ft^2 default is the ASHRAE 62.1 office value as
    reported in secondary sources. Verify it against the edition you cite.
    """
    area_ft2 = area_m2 / 0.09290304
    return density_per_1000ft2 * area_ft2 / 1000.0


def breathing_zone_flow_ls(
    area_m2: float,
    people: float,
    rp_ls_per_person: float = 2.5,
    ra_ls_per_m2: float = 0.3,
) -> float:
    """Minimum breathing-zone outdoor airflow, ``Vbz = Rp*Pz + Ra*Az`` (L/s).

    Defaults are the office-space rates (2.5 L/s per person, 0.3 L/s per m^2).
    This is a lower bound: it excludes zone air distribution effectiveness and
    system ventilation efficiency. Verify the rates against the standard
    edition you cite.

    Args:
        area_m2: Floor area served (m^2).
        people: Number of occupants (an assumption: the dataset has no headcount).
        rp_ls_per_person: People component rate (L/s per person).
        ra_ls_per_m2: Area component rate (L/s per m^2).

    Returns:
        Required outdoor airflow in L/s.
    """
    return rp_ls_per_person * people + ra_ls_per_m2 * area_m2
