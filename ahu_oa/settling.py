"""Fan start-up settling: drop rows soon after the supply fan starts.

The mixed air temperature is only representative once air has been moving
for a while, so rows in the first minutes of each fan run are not trusted.
"""
from __future__ import annotations

import pandas as pd


def settled_mask(index: pd.DatetimeIndex, on: pd.Series, settle: str = "10min") -> pd.Series:
    """Return True for fan-on rows at least ``settle`` after their run began.

    Each continuous run of ``on == True`` is timed from its first row. Compute
    this on the full dataframe, before any rows are dropped, so every start-up
    is still visible.

    Args:
        index: Sorted ``DatetimeIndex`` of the data.
        on: Boolean series (True while the fan is on), aligned with ``index``.
        settle: Pandas timedelta string, for example ``"10min"``.

    Returns:
        Boolean series aligned with ``index``.

    Raises:
        TypeError: If ``index`` is not a ``DatetimeIndex``.
        ValueError: If ``index`` is not sorted in increasing order.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("settled_mask needs a DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError("settled_mask needs a time-sorted index")
    on = pd.Series(on.to_numpy(bool), index=index)
    run = on.ne(on.shift(fill_value=False)).cumsum()
    t = index.to_series()
    start = t.groupby(run).transform("min")
    return on & ((t - start) >= pd.Timedelta(settle))
