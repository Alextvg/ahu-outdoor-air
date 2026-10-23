"""Cubic spline estimator of damper position from the outdoor air fraction.

Method (spline interpolation): the damper position is modelled as a smooth
function of the fraction ``f`` by a natural cubic spline.

1. Keep well-conditioned rows with a finite ``f_phys`` and damper position.
2. Cut ``f`` into ``n_knots`` bins and summarise each bin by its median ``f`` and
   median damper position. The median resists the few transient rows where ``f``
   is high while the damper is nearly closed, and one point per bin averages out
   sensor noise.
3. Fit a natural cubic spline through those knot points. Between two knots the
   curve is a cubic polynomial; the pieces join with continuous value, slope and
   curvature, and the curvature is zero at the two ends (natural boundary).

Outside the range of training fractions the prediction is held flat at the end
knot, and predictions are limited to the range of the knot positions. A cubic
spline can overshoot between knots, so this keeps the output physical (inside the
range seen in training) but the curve is not guaranteed to be monotone. Fewer
knots give a smoother curve; the number is chosen by the error on the validation block.

The fitted object has the same interface as :class:`ahu_oa.damper_model.DamperEstimator`
(``name``, ``kind``, ``params``, ``n_train``, ``predict``), so the scoring code in
``scripts/run_damper_model.py`` can use it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

KNOT_RULES = ("quantile", "uniform")
# Log-spaced knot counts. Small values show how badly a coarse spline fits, the middle
# values bracket where the error usually flattens, and the top is large enough for the
# curve to follow the S-shape without every bin holding only a few rows.
DEFAULT_KNOT_CANDIDATES = (6, 10, 15, 20, 30, 45, 60, 90, 120, 180, 250)
UNIFORM_TRIM = (0.5, 99.5)  # percentiles bounding the uniform knot range


@dataclass
class SplineEstimator:
    """A fitted natural cubic spline from outdoor air fraction to damper position.

    Attributes:
        name: Label used in result tables.
        kind: Always ``"spline"``.
        params: ``{"x": knot fractions, "y": knot positions, "n_knots": requested
            number of bins, "rule": knot rule}``. ``x`` is strictly increasing. The actual number of knots can be smaller than
            ``n_knots`` if bins merge.
        n_train: Rows used in the fit.
    """

    name: str
    kind: str
    params: dict
    n_train: int

    def predict(self, f) -> np.ndarray:
        """Damper position implied by fractions ``f`` (NaN stays NaN).

        Fractions outside the knot range are held at the end knot, and the result
        is limited to the range of the knot positions, so a spline overshoot
        between knots cannot give an impossible position.
        """
        f = np.asarray(f, dtype=float)
        x = np.asarray(self.params["x"], dtype=float)
        y = np.asarray(self.params["y"], dtype=float)
        out = np.full(f.shape, np.nan)
        ok = np.isfinite(f)
        if ok.any():
            spline = CubicSpline(x, y, bc_type="natural")
            out[ok] = np.clip(spline(np.clip(f[ok], x[0], x[-1])), y.min(), y.max())
        return out


def _bin_edges(f: np.ndarray, n_knots: int, rule: str) -> np.ndarray:
    """Bin edges for ``f``: equal-count (quantile) or equal-width (uniform)."""
    if rule == "quantile":
        return np.unique(np.quantile(f, np.linspace(0.0, 1.0, n_knots + 1)))
    lo, hi = np.percentile(f, UNIFORM_TRIM)
    return np.linspace(lo, hi, n_knots + 1)


def fit_spline(
    train: pd.DataFrame,
    damper_col: str,
    n_knots: int = 20,
    rule: str = "quantile",
) -> SplineEstimator:
    """Fit the natural cubic spline ``damper = g(f_phys)``.

    Args:
        train: Dataframe with ``f_phys``, ``well_conditioned`` and ``damper_col``.
        damper_col: Column holding the true damper position.
        n_knots: Number of ``f`` bins, so up to this many spline knots. At least 3. Too many knots
            lets the curve follow noise and wiggle.
        rule: ``"quantile"`` gives equal row counts per bin, so knots crowd where
            the data are dense (low fractions, damper near its minimum).
            ``"uniform"`` gives equal-width bins between the 0.5th and 99.5th
            percentile of ``f``, so sparse regions get knots too.

    Returns:
        A :class:`SplineEstimator`.

    Raises:
        ValueError: On an unknown ``rule``, ``n_knots < 3``, or when the data
            give fewer than 3 distinct knots.
    """
    if rule not in KNOT_RULES:
        raise ValueError(f"unknown rule {rule!r}; use one of {KNOT_RULES}")
    if n_knots < 3:
        raise ValueError("n_knots must be at least 3")
    d = train[train["well_conditioned"]][["f_phys", damper_col]].dropna()
    if len(d) < 3 * n_knots:
        raise ValueError(f"need at least {3 * n_knots} rows for {n_knots} knots, got {len(d)}")
    f = d["f_phys"].to_numpy(float)
    y = d[damper_col].to_numpy(float)

    edges = _bin_edges(f, n_knots, rule)
    # Interior edges only; rows beyond the ends fall into the first or last bin.
    bin_id = np.digitize(f, edges[1:-1])
    per_bin = (pd.DataFrame({"f": f, "y": y, "bin": bin_id})
               .groupby("bin")
               .agg(x=("f", "median"), y=("y", "median"), w=("f", "size"))
               .reset_index(drop=True))
    # Bins with an identical median f would give a repeated knot: merge them,
    # weighting their positions by row count.
    per_bin["wy"] = per_bin["y"] * per_bin["w"]
    knots = per_bin.groupby("x", as_index=False).agg(wy=("wy", "sum"), w=("w", "sum"))
    knots["y"] = knots["wy"] / knots["w"]
    knots = knots.sort_values("x")
    if len(knots) < 3:
        raise ValueError(f"only {len(knots)} distinct knots; the fractions barely vary")

    x_k = knots["x"].to_numpy(float)
    y_k = knots["y"].to_numpy(float)
    return SplineEstimator(
        name=f"cubic_spline_{rule}_{len(x_k)}", kind="spline",
        params={"x": [float(v) for v in x_k], "y": [float(v) for v in y_k],
                "n_knots": int(n_knots), "rule": rule},
        n_train=len(d))


def select_n_knots(
    train: pd.DataFrame,
    val: pd.DataFrame,
    damper_col: str,
    candidates=DEFAULT_KNOT_CANDIDATES,
    rule: str = "quantile",
    tolerance: float = 0.05,
) -> tuple[int, pd.DataFrame]:
    """Choose the number of knots by the error on the validation block.

    For each candidate, fit the spline on ``train`` and score it on ``val``. The
    error usually falls quickly and then flattens, so the lowest error alone tends
    to pick the largest candidate for a negligible gain. The chosen count is the
    smallest one whose validation MAE is within ``tolerance`` of the best, which is
    the simplest curve that is almost as accurate. The test block and the stuck
    files are not used here, so the final results are not tuned to them.

    Args:
        train: Training dataframe (see :func:`fit_spline`).
        val: Validation dataframe with ``f_phys``, ``well_conditioned`` and
            ``damper_col``. Only well-conditioned rows are scored.
        damper_col: Column holding the true damper position.
        candidates: Knot counts to try.
        rule: Knot rule passed to :func:`fit_spline`.
        tolerance: Relative margin over the lowest MAE that still counts as good
            enough (0.05 is 5%). Use 0 for the plain lowest-error choice.

    Returns:
        Tuple ``(chosen_n_knots, table)``. ``table`` has one row per candidate with
        its validation MAE, the number of rows scored (NaN MAE if the fit failed)
        and ``within_tolerance``. A warning is issued if the chosen count is the
        smallest or largest that could be fitted, since the flat part of the error
        curve may then lie outside the grid.

    Raises:
        ValueError: If no candidate could be fitted and scored, or ``tolerance < 0``.
    """
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")
    held = val[val["well_conditioned"]]
    truth = held[damper_col].to_numpy(float)
    rows = []
    for k in candidates:
        try:
            est = fit_spline(train, damper_col, n_knots=k, rule=rule)
        except ValueError:
            rows.append({"n_knots": k, "val_mae": np.nan, "n_scored": 0})
            continue
        err = np.abs(est.predict(held["f_phys"].to_numpy(float)) - truth)
        err = err[np.isfinite(err)]
        rows.append({"n_knots": k, "val_mae": float(err.mean()) if err.size else np.nan,
                     "n_scored": int(err.size)})
    table = pd.DataFrame(rows)
    if table["val_mae"].isna().all():
        raise ValueError("no candidate could be scored; check the data and candidates")
    limit = table["val_mae"].min() * (1.0 + tolerance)
    table["within_tolerance"] = table["val_mae"] <= limit
    chosen = int(table.loc[table["within_tolerance"], "n_knots"].min())
    fitted = table.dropna(subset=["val_mae"])["n_knots"]
    if len(fitted) > 1 and chosen in (fitted.min(), fitted.max()):
        warnings.warn(f"chosen knot count {chosen} is at the edge of the candidates "
                      f"({fitted.min()} to {fitted.max()}); widen the range", stacklevel=2)
    return chosen, table
