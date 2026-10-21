"""Load LBNL single-duct AHU CSV files, convert to SI and audit data quality.

Column names follow the dataset documentation (Granderson et al., 2022).
NOTE: the name of the timestamp column has not been verified against the real
CSV files. Pass ``time_col`` once you have checked it.
"""
from __future__ import annotations

import pandas as pd

from ahu_oa import units

TEMP_COLS = [
    "SA_TEMP", "SA_TEMPSPT", "OA_TEMP", "MA_TEMP", "RA_TEMP",
    "ZONE_TEMP_1", "ZONE_TEMP_2", "ZONE_TEMP_3", "ZONE_TEMP_4", "ZONE_TEMP_5",
]
FLOW_COLS = ["OA_CFM", "RA_CFM", "SA_CFM"]
PRESSURE_COLS = ["SA_SP", "SA_SPSPT"]
# Signals documented as ranging from 0 to 1 (fan speeds, dampers, valve).
FRACTION_COLS = [
    "SF_CS", "SF_SPD", "RF_CS", "RF_SPD",
    "OA_DMPR_DM", "OA_DMPR", "RA_DMPR_DM", "RA_DMPR",
    "CHWC_VLV_DM", "CHWC_VLV",
]


def convert_to_si(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataset columns to SI and add unit suffixes to their names.

    Temperatures become ``*_C`` (degC), flows ``*_Ls`` (L/s) and pressures
    ``*_Pa`` (Pa). Columns that are not present are skipped, and all other
    columns are left unchanged.

    Args:
        df: Raw dataframe using the original dataset column names and units.

    Returns:
        A new dataframe in SI units.
    """
    out = df.copy()
    renames = {}
    for col in TEMP_COLS:
        if col in out:
            out[col] = units.f_to_c(out[col])
            renames[col] = f"{col}_C"
    for col in FLOW_COLS:
        if col in out:
            out[col] = units.cfm_to_ls(out[col])
            renames[col] = col.replace("_CFM", "_Ls")
    for col in PRESSURE_COLS:
        if col in out:
            out[col] = units.inwg_to_pa(out[col])
            renames[col] = f"{col}_Pa"
    return out.rename(columns=renames)


def load_ahu_csv(path: str, time_col: str | None = "Datetime") -> pd.DataFrame:
    """Read one AHU CSV file and return it in SI units.

    Args:
        path: Path to a CSV file such as ``data/AHU_annual.csv``.
        time_col: Name of the timestamp column, if known. When given it is
            parsed as datetimes and used as the index.

    Returns:
        Dataframe in SI units (see :func:`convert_to_si`).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    return convert_to_si(df)


def occupied_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask that is True when the system is in occupied mode.

    Uses the ``SYS_CTL`` indicator (1 = occupied, 0 = unoccupied).
    """
    return df["SYS_CTL"] == 1


def audit_ahu(df: pd.DataFrame, tol: float = 1e-6) -> dict:
    """Run basic data-quality checks and report problems (nothing is dropped).

    Args:
        df: Dataframe from :func:`load_ahu_csv`.
        tol: Tolerance when checking that 0-1 signals stay in range.

    Returns:
        Dictionary with the row count, columns containing missing values,
        columns that never change, and 0-1 signals that leave their range.
    """
    missing = df.isna().sum()
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    out_of_range = {}
    for col in FRACTION_COLS:
        if col in df:
            n_bad = int(((df[col] < -tol) | (df[col] > 1 + tol)).sum())
            if n_bad:
                out_of_range[col] = n_bad
    return {
        "n_rows": len(df),
        "missing_values": missing[missing > 0].to_dict(),
        "constant_columns": constant,
        "out_of_range_0_1": out_of_range,
    }
