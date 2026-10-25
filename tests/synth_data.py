"""Synthetic AHU CSV builder shared by the audit and script tests."""
import numpy as np
import pandas as pd


def build_csv(path, stuck=None, start="2020-01-01", end="2020-12-31 23:00", seed=0,
              label_pos=None, constant_col=True, drop_time=False, cap_damper=None):
    """Write a raw-unit AHU CSV (deg F, CFM) and return its path as a string.

    Args:
        stuck: True damper position when the damper is stuck, else None.
        label_pos: Position recorded in OA_DMPR when it differs from the truth.
        constant_col: If False the recorded stuck position is noisy.
        cap_damper: Upper limit on the fault-free damper (to test coverage).
    """
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="1h")
    n = len(t)
    oa = 55 + 30 * np.sin(2 * np.pi * (t.dayofyear.values - 110) / 365) \
        + 8 * np.sin(2 * np.pi * t.hour.values / 24)
    dm = np.clip(0.5 + 0.45 * np.sin(np.arange(n) / 37) + 0.1 * np.sin(np.arange(n) / 5), 0, 1)
    if cap_damper is not None:
        dm = np.minimum(dm, cap_damper)
    pos = dm.copy() if stuck is None else np.full(n, stuck)
    f = 0.1 + 0.6 * pos
    ma = 72.0 + f * (oa - 72.0) + rng.normal(0, 0.2, n)
    shown = pos if stuck is None else np.full(n, label_pos if label_pos is not None else stuck)
    if stuck is not None and not constant_col:
        shown = np.clip(shown + rng.normal(0, 0.2, n), 0, 1)
    df = pd.DataFrame({
        "Datetime": t, "OA_TEMP": oa, "RA_TEMP": 72.0, "MA_TEMP": ma,
        "SF_SPD_DM": (rng.random(n) > 0.1).astype(int), "SYS_CTL": 1,
        "SF_CS": rng.uniform(0.5, 0.9, n),
        "OA_DMPR_DM": dm, "OA_DMPR": shown, "SA_CFM": 2000.0, "OA_CFM": 500.0})
    if drop_time:
        df = df.drop(columns="Datetime")
    df.to_csv(path, index=False)
    return str(path)
