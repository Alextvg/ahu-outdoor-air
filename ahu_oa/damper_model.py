"""Estimate damper position from the outdoor air fraction (inverse model).

Design: the mixing equation gives an outdoor air fraction ``f`` from three
temperatures. On fault-free data the damper position and ``f`` are related;
that relationship is learned and inverted, so a new row's ``f`` implies a
damper position. Applied to a file where the damper is stuck at a known
position, the implied position should come out at that position.

Two estimators are provided:

* ``inverted_linearised``: fit ``f = a + b * damper`` with the linearised
  regression (no division by small temperature gaps), then invert it.
* ``direct``: fit ``damper`` as a polynomial in ``f`` on well-conditioned
  rows. This handles noise in ``f`` and mild curvature directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ahu_oa import regression


@dataclass
class DamperEstimator:
    """A fitted map from outdoor air fraction to damper position.

    Attributes:
        name: Label used in result tables.
        kind: ``"inverted"`` or ``"direct"``.
        params: ``{"a", "b"}`` for inverted; ``{"coefs"}`` (highest power
            first) for direct.
        n_train: Rows used in the fit.
    """

    name: str
    kind: str
    params: dict
    n_train: int

    def predict(self, f) -> np.ndarray:
        """Damper position implied by fractions ``f`` (NaN stays NaN).

        Not clipped to 0-1, so out-of-range answers stay visible.
        """
        f = np.asarray(f, dtype=float)
        if self.kind == "inverted":
            a, b = self.params["a"], self.params["b"]
            return (f - a) / b if b > 0 else np.full_like(f, np.nan)
        if self.kind == "direct":
            return np.polyval(self.params["coefs"], f)
        raise ValueError(f"unknown kind {self.kind!r}")


def fit_inverted(train: pd.DataFrame, damper_col: str) -> DamperEstimator:
    """Fit ``f = a + b*damper`` on well-conditioned rows and invert it.

    Args:
        train: Filtered dataframe with the three temperatures, the damper
            column and a ``well_conditioned`` flag.
        damper_col: Damper column holding the true position.

    Returns:
        A :class:`DamperEstimator`.
    """
    m = regression.fit(train, damper_col=damper_col, well_conditioned_only=True)
    return DamperEstimator("inverted_linearised", "inverted",
                           {"a": m.coefs["a"], "b": m.coefs["b"]}, m.n_train)


def fit_direct(train: pd.DataFrame, damper_col: str, degree: int = 2) -> DamperEstimator:
    """Fit damper position as a polynomial in ``f_phys`` on well-conditioned rows.

    Args:
        train: Dataframe with ``f_phys``, ``well_conditioned`` and ``damper_col``.
        damper_col: Damper column holding the true position.
        degree: Polynomial degree (1 = straight line).

    Raises:
        ValueError: If there are not enough rows for the degree.
    """
    d = train[train["well_conditioned"]][["f_phys", damper_col]].dropna()
    if len(d) <= degree + 1:
        raise ValueError(f"need more than {degree + 1} rows, got {len(d)}")
    coefs = np.polyfit(d["f_phys"].to_numpy(float), d[damper_col].to_numpy(float), degree)
    return DamperEstimator(f"direct_poly{degree}", "direct",
                           {"coefs": [float(c) for c in coefs]}, len(d))
