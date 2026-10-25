"""Single place for paths and tunable settings used by the scripts.

Scripts import this (``import config``) so a value such as the ill-conditioning
threshold is changed once, not in every script. Run scripts from the project
root with ``python -m scripts.<name>`` so that ``config`` and ``ahu_oa`` import.
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # raw CSVs are NOT committed (see data/README.md)
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---- Data files ------------------------------------------------------------
# Case label -> CSV file name. The label is used in tables and figure names.
FILES = {
    "fault_free": "AHU_annual.csv",
    "oa_bias_+2": "oa_bias_2_annual.csv",
    "oa_bias_+4": "oa_bias_4_annual.csv",
    "oa_bias_-2": "oa_bias_-2_annual.csv",
    "oa_bias_-4": "oa_bias_-4_annual.csv",
}
# Outdoor air temperature sensor bias in each case (K). Documentation only.
BIAS_K = {
    "fault_free": 0.0,
    "oa_bias_+2": 2.0, "oa_bias_+4": 4.0,
    "oa_bias_-2": -2.0, "oa_bias_-4": -4.0,
}

# Name of the timestamp column. NOT verified against the real CSVs: run
# ``python -m scripts.run_audit`` first, which lists the real column names.
TIME_COL = "Datetime"

# ---- Filtering and baseline settings ---------------------------------------
OCCUPIED_RULE = "sys_ctl"   # or "fan_status"; compare both in the audit
MIN_DELTA_K = 5.56           # |T_outdoor - T_return| below this is "ill-conditioned"

# ---- Compliance check (ASHRAE 62.1 ventilation rate procedure) -------------
FLOOR_AREA_M2 = 3563.0      # from the dataset documentation (check it)
PEOPLE_PER_1000FT2 = 5.0    # default office density; an assumption (state it)

# ---- Damper-stuck design ---------------------------------------------------
# Used by scripts.run_damper_audit and scripts.run_damper_model. The idea:
# learn damper position from the mixing equation on fault-free data, then test
# on files where the damper is stuck at a known position.
STUCK_FILES = {
    "damper_stuck_010": "damper_stuck_010_annual.csv",
    "damper_stuck_025": "damper_stuck_025_annual.csv",
    "damper_stuck_075": "damper_stuck_075_annual.csv",
    "damper_stuck_100": "damper_stuck_100_annual_short.csv",
}
# Known damper position (0-1) for each stuck case. The audit checks it against
# the data, so a wrong entry here is reported, not silently used.
STUCK_POSITION = {
    "damper_stuck_010": 0.10,
    "damper_stuck_025": 0.25,
    "damper_stuck_075": 0.75,
    "damper_stuck_100": 1.00,
}
# Damper column holding the true position (OA_DMPR = measured position,
# OA_DMPR_DM = command). The audit shows which one is constant when stuck.
DAMPER_COL = "OA_DMPR"

# The mixing equation only holds while air is moving, so this design always
# needs fan-on rows (OCCUPIED_RULE above stays as it is for the other scripts).
DAMPER_OCCUPIED_RULE = "fan_status"
SETTLE = "10min"             # fan-on time before mixed air is trusted (tune on data)
SPLIT_METHOD = "blocked"     # "blocked" or "chronological" (fault-free train/val/test)
DIRECT_POLY_DEGREE = 2       # degree of the direct damper-from-fraction estimator
POSITION_TOL = 0.05          # |implied - known| damper position counted as agreement
