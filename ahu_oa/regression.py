"""Linearised regression model for the outdoor air fraction.

The mixing equation is rearranged so that no division is needed:

    (T_mixed - T_return) = (a + b*damper [+ c*fan [+ d*damper*fan]]) * (T_outdoor - T_return)

This is linear in the coefficients, so ordinary least squares applies. There
is no separate intercept: the bracket multiplies the temperature difference.
Rows where the difference is small simply contribute little to the fit.

Once fitted, the outdoor air fraction is the bracket alone, so predicting it
needs only the damper (and fan) signal, never the temperatures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TEMP_COLS = ("MA_TEMP_C", "RA_TEMP_C", "OA_TEMP_C")
COND_WARN = 1e3  # condition number above which the design is flagged


def _terms(damper, fan, use_fan: bool, interaction: bool):
    """Build the bracket terms ``[1, damper, fan, damper*fan]`` as needed."""
    cols = [np.ones_like(damper), damper]
    names = ["a", "b"]
    if use_fan:
        cols.append(fan)
        names.append("c")
        if interaction:
            cols.append(damper * fan)
            names.append("d")
    return np.column_stack(cols), names


@dataclass
class FractionModel:
    """A fitted linearised model.

    Attributes:
        coefs: Fitted coefficients by name (``a``, ``b``, and ``c``/``d`` if
            the fan terms are used).
        stderr: Standard error of each coefficient (NaN if not computable).
        use_fan: Whether the fan signal is a regressor.
        interaction: Whether the damper-by-fan term is included.
        damper_col: Damper signal column.
        fan_col: Fan signal column.
        n_train: Rows used in the fit.
        cond: Condition number of the column-scaled design matrix. Large
            values mean collinear regressors and unstable coefficients.
        rmse_K: Residual RMSE of ``T_mixed - T_return`` in kelvin.
    """

    coefs: dict
    stderr: dict
    use_fan: bool
    interaction: bool
    damper_col: str
    fan_col: str
    n_train: int
    cond: float
    rmse_K: float

    def terms(self, df: pd.DataFrame) -> np.ndarray:
        """Bracket-term matrix for ``df`` (NaN where a signal is missing)."""
        damper = df[self.damper_col].to_numpy(float)
        fan = df[self.fan_col].to_numpy(float) if self.use_fan else np.zeros(len(df))
        T, _ = _terms(damper, fan, self.use_fan, self.interaction)
        return T

    def predict_fraction(self, df: pd.DataFrame) -> np.ndarray:
        """Outdoor air fraction from the damper (and fan) signal only."""
        return self.terms(df) @ np.array(list(self.coefs.values()))

    def predict_flow(self, df: pd.DataFrame, supply_col: str = "SA_Ls") -> np.ndarray:
        """Outdoor airflow (L/s) = fraction x supply airflow."""
        return self.predict_fraction(df) * df[supply_col].to_numpy(float)

    def warnings(self) -> list[str]:
        """Plausibility checks on the fitted coefficients."""
        out = []
        if self.cond > COND_WARN:
            out.append(f"design matrix ill-conditioned (cond={self.cond:.3g})")
        a, b = self.coefs["a"], self.coefs["b"]
        if b <= 0:
            out.append(f"b={b:.3g} is not positive: opening the damper should raise the fraction")
        if a < -0.05:
            out.append(f"a={a:.3g} is negative")
        if not self.use_fan and a + b > 1.1:
            out.append(f"a+b={a + b:.3g} exceeds 1: fully open damper gives more than 100% outdoor air")
        return out


def fit(
    df: pd.DataFrame,
    use_fan: bool = False,
    interaction: bool = False,
    damper_col: str = "OA_DMPR_DM",
    fan_col: str = "SF_CS",
    well_conditioned_only: bool = False,
) -> FractionModel:
    """Fit the linearised model by least squares.

    Args:
        df: Filtered dataframe (SI names). Rows with a missing needed value
            are dropped.
        use_fan: Add a ``c * fan`` term to the bracket.
        interaction: Add a ``d * damper * fan`` term (needs ``use_fan``).
        damper_col: Damper signal. Use ``"OA_DMPR"`` for measured position.
        fan_col: Fan signal.
        well_conditioned_only: Fit only on rows flagged ``well_conditioned``.

    Returns:
        A :class:`FractionModel`.

    Raises:
        ValueError: On an invalid option combination or too few rows.
        KeyError: If a needed column is missing.
    """
    if interaction and not use_fan:
        raise ValueError("interaction needs use_fan=True")
    need = list(TEMP_COLS) + [damper_col] + ([fan_col] if use_fan else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    d = df
    if well_conditioned_only:
        if "well_conditioned" not in d.columns:
            raise KeyError("well_conditioned column missing (run apply_filters first)")
        d = d[d["well_conditioned"]]
    d = d[need].dropna()

    dT = (d["OA_TEMP_C"] - d["RA_TEMP_C"]).to_numpy(float)
    y = (d["MA_TEMP_C"] - d["RA_TEMP_C"]).to_numpy(float)
    damper = d[damper_col].to_numpy(float)
    fan = d[fan_col].to_numpy(float) if use_fan else np.zeros(len(d))
    T, names = _terms(damper, fan, use_fan, interaction)
    k = len(names)
    n = len(d)
    if n <= k:
        raise ValueError(f"need more than {k} rows to fit, got {n}")

    X = T * dT[:, None]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    norms = np.linalg.norm(X, axis=0)
    cond = float("inf") if np.any(norms == 0) else float(np.linalg.cond(X / norms))

    return FractionModel(
        coefs=dict(zip(names, map(float, coef))),
        stderr=dict(zip(names, map(float, se))),
        use_fan=use_fan,
        interaction=interaction,
        damper_col=damper_col,
        fan_col=fan_col,
        n_train=n,
        cond=cond,
        rmse_K=float(np.sqrt(resid @ resid / n)),
    )
