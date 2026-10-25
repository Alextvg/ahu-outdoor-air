"""Train / validation / test splits that avoid leaking neighbouring rows.

Adjacent time steps are nearly identical, so a random row split would put
almost the same row in train and test and flatter every model. Splits here
are made in contiguous time blocks instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ("train", "val", "test")


def split_data(
    df: pd.DataFrame,
    method: str = "blocked",
    fractions=(0.6, 0.2, 0.2),
    block: str = "7D",
    pattern=("train", "train", "train", "val", "test"),
) -> dict:
    """Split a time-indexed dataframe into train, val and test parts.

    Args:
        df: Dataframe with a sorted ``DatetimeIndex``.
        method: ``"blocked"`` cuts the timeline into ``block``-long blocks and
            deals them out following ``pattern`` (cycled). Every season is then
            represented in every part, which suits an annual dataset.
            ``"chronological"`` uses the first ``fractions[0]`` of the rows for
            train, the next for val, and the rest for test, so the test period
            is a different season from training.
        fractions: Row shares for the chronological method (sum to 1).
        block: Block length for the blocked method (pandas timedelta string).
        pattern: Label for each consecutive block, cycled.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to dataframes.

    Raises:
        TypeError: If the index is not a ``DatetimeIndex``.
        ValueError: If the index is unsorted, the method is unknown, or a part
            ends up empty.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("split_data needs a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("split_data needs a time-sorted index")

    if method == "chronological":
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("fractions must sum to 1")
        n = len(df)
        i1 = int(round(n * fractions[0]))
        i2 = int(round(n * (fractions[0] + fractions[1])))
        parts = {"train": df.iloc[:i1], "val": df.iloc[i1:i2], "test": df.iloc[i2:]}
    elif method == "blocked":
        if not set(pattern) <= set(LABELS):
            raise ValueError(f"pattern labels must be from {LABELS}")
        block_id = np.asarray((df.index - df.index.min()) // pd.Timedelta(block), dtype=int)
        label = np.array(pattern)[block_id % len(pattern)]
        parts = {k: df[label == k] for k in LABELS}
    else:
        raise ValueError(f"unknown method {method!r}")

    empty = [k for k, v in parts.items() if v.empty]
    if empty:
        raise ValueError(f"split produced empty parts: {empty}")
    return parts
