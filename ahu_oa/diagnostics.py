"""Checks on how the regression behaves, to run before trusting it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ahu_oa import regression


def _row(label, model):
    return {"fit": label, "n_train": model.n_train, "cond": model.cond,
            "rmse_K": model.rmse_K, **model.coefs}


def compare_all_vs_flagged(df: pd.DataFrame, **fit_kwargs) -> pd.DataFrame:
    """Fit on all rows and on well-conditioned rows only, side by side.

    If the coefficients barely move, the delta threshold does not matter for
    the regression and all rows can be used for fitting.
    """
    all_m = regression.fit(df, well_conditioned_only=False, **fit_kwargs)
    flg_m = regression.fit(df, well_conditioned_only=True, **fit_kwargs)
    return pd.DataFrame([_row("all_rows", all_m), _row("well_conditioned_only", flg_m)])


def coefficient_stability(
    df: pd.DataFrame, by: str = "month", min_rows: int = 50, **fit_kwargs
) -> pd.DataFrame:
    """Fit separately per group and show how the coefficients move.

    Args:
        df: Filtered dataframe with a ``DatetimeIndex``.
        by: ``"month"``, or ``"half"`` (first and second half of the rows).
        min_rows: Groups with fewer rows are skipped.
        **fit_kwargs: Passed to :func:`regression.fit`.
    """
    if by == "month":
        groups = df.groupby(df.index.month)
    elif by == "half":
        half = np.where(np.arange(len(df)) < len(df) // 2, "first", "second")
        groups = df.groupby(half)
    else:
        raise ValueError("by must be 'month' or 'half'")
    rows = []
    for key, g in groups:
        if len(g) < min_rows:
            continue
        try:
            m = regression.fit(g, **fit_kwargs)
        except ValueError:
            continue
        rows.append({"group": key, **_row(by, m)})
    return pd.DataFrame(rows).drop(columns="fit", errors="ignore")


def fan_variation(df: pd.DataFrame, fan_col: str = "SF_CS", min_std: float = 0.02) -> dict:
    """How much the fan signal varies in the filtered rows.

    A (near-)constant fan signal is collinear with the intercept ``a`` and
    carries no information, so the fan term should be dropped.

    Args:
        min_std: Standard deviation below which the signal counts as constant.
            A tunable choice.
    """
    s = df[fan_col].dropna()
    return {"min": float(s.min()), "max": float(s.max()), "std": float(s.std()),
            "n_unique": int(s.nunique()), "varies": bool(s.std() >= min_std)}


def residual_by_damper(
    df: pd.DataFrame, model: regression.FractionModel,
    f_ref_col: str = "f_phys", bins: int = 10,
) -> pd.DataFrame:
    """Residual of the fitted fraction against damper position.

    Compares the model's fraction with the physics fraction on well-conditioned
    rows, grouped by damper position. A systematic arch means the straight
    line in the damper is too simple.
    """
    d = df[df["well_conditioned"]].copy()
    d["f_hat"] = model.predict_fraction(d)
    d["resid"] = d[f_ref_col] - d["f_hat"]
    d = d.dropna(subset=["resid", model.damper_col])
    d["damper_bin"] = pd.cut(d[model.damper_col], bins=bins)
    g = d.groupby("damper_bin", observed=True)["resid"]
    return pd.DataFrame({"n": g.size(), "mean_resid": g.mean(), "std_resid": g.std()})
