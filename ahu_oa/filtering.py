"""Row filters applied after loading, before any modelling or scoring.

Loading only converts units. This module decides which rows are used:

1. occupied rows only (fans on, damper not closed by design),
2. rows with no missing values in the signals the models need,
3. a flag (not a drop) for rows where outdoor and return air temperatures are
   too close for the mixing equation to be trusted.

Column names are the SI names produced by ``loading.load_ahu_csv``.
"""
from __future__ import annotations

import pandas as pd

# Signals the outdoor air models need (names after SI conversion).
REQUIRED_COLS = [
    "OA_TEMP_C", "RA_TEMP_C", "MA_TEMP_C",
    "SF_CS", "OA_DMPR_DM", "SA_Ls", "OA_Ls",
]


def occupied_mask(df: pd.DataFrame, rule: str = "sys_ctl") -> pd.Series:
    """Return True for rows in occupied operation.

    Args:
        df: Dataframe from ``loading.load_ahu_csv``.
        rule: ``"sys_ctl"`` uses the occupancy flag (``SYS_CTL == 1``), which is
            not commonly logged in real buildings. ``"fan_status"`` uses the
            supply fan status (``SF_SPD_DM == 1``), a commonly logged point.
            Compare the two on your data before choosing.

    Returns:
        Boolean series aligned with ``df``.
    """
    if rule == "sys_ctl":
        return df["SYS_CTL"] == 1
    if rule == "fan_status":
        return df["SF_SPD_DM"] == 1
    raise ValueError(f"Unknown rule {rule!r}; use 'sys_ctl' or 'fan_status'")


def valid_mask(df: pd.DataFrame, required_cols=None) -> pd.Series:
    """Return True for rows with no missing values in the required columns.

    Missing values are excluded, not imputed. Out-of-range values are reported
    by ``loading.audit_ahu`` and are not removed here.
    """
    cols = REQUIRED_COLS if required_cols is None else required_cols
    return df[cols].notna().all(axis=1)


def well_conditioned_mask(df: pd.DataFrame, min_delta: float = 3.0) -> pd.Series:
    """Return True where ``|T_outdoor - T_return| >= min_delta`` (K).

    Uses the *measured* temperatures, as an engineer in a real building would.
    In the sensor-bias files the biased outdoor reading changes this mask.
    ``min_delta`` is a tunable placeholder; tune it on validation data only.
    """
    return (df["OA_TEMP_C"] - df["RA_TEMP_C"]).abs() >= min_delta


def apply_filters(
    df: pd.DataFrame,
    occupied_rule: str = "sys_ctl",
    required_cols=None,
    min_delta: float = 3.0,
):
    """Keep occupied, complete rows and flag the well-conditioned ones.

    Rows are dropped for being unoccupied or incomplete. Poorly conditioned
    rows are kept but flagged in a new boolean column ``well_conditioned``, so
    both models can be scored on the same rows while the excluded rows are
    still available to report on separately.

    Args:
        df: Dataframe from ``loading.load_ahu_csv``.
        occupied_rule: See :func:`occupied_mask`.
        required_cols: Columns that must be non-missing.
        min_delta: See :func:`well_conditioned_mask`.

    Returns:
        Tuple ``(filtered_df, report)``. ``report`` holds the row counts after
        each stage and the share of occupied rows that are well conditioned.
    """
    n_input = len(df)
    occ = occupied_mask(df, occupied_rule)
    after_occ = df[occ]
    ok = valid_mask(after_occ, required_cols)
    out = after_occ[ok].copy()
    out["well_conditioned"] = well_conditioned_mask(out, min_delta)

    n_out = len(out)
    report = {
        "n_input": n_input,
        "n_occupied": int(occ.sum()),
        "n_complete": n_out,
        "n_well_conditioned": int(out["well_conditioned"].sum()),
        "coverage_well_conditioned": (
            float(out["well_conditioned"].mean()) if n_out else float("nan")
        ),
        "occupied_rule": occupied_rule,
        "min_delta": min_delta,
    }
    return out, report
