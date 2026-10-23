"""Plot helpers. They draw on a given axes and return it; they never save or show.

The calling script owns the figure, so it decides whether to save it to
``results/figures`` or display it in a notebook. Dense one-minute data is drawn
as hexagonal bins, which stay readable with hundreds of thousands of rows.
"""
from __future__ import annotations

import numpy as np


def _finite(*arrays):
    """Return the arrays restricted to rows where every array is finite."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    ok = np.logical_and.reduce([np.isfinite(a) for a in arrays])
    return [a[ok] for a in arrays]


def plot_est_vs_true(ax, y_true, y_pred, xlabel="True", ylabel="Estimated",
                     title=None, one_to_one=True):
    """Density plot of one quantity against another, optionally with a 1:1 line.

    Used for estimate-vs-truth (both axes the same quantity, ``one_to_one=True``)
    and for relationships between different quantities such as fan command vs
    airflow (``one_to_one=False``: axes scale independently, no reference line).

    Args:
        ax: Matplotlib axes to draw on.
        y_true: Values for the x axis (ground truth).
        y_pred: Values for the y axis (estimate). Non-finite pairs are skipped.
        xlabel: Label for the x axis (include units).
        ylabel: Label for the y axis (include units).
        title: Optional axes title.
        one_to_one: Share one range on both axes and draw a 1:1 line. Only
            meaningful when both axes have the same units.

    Returns:
        The axes, for chaining.
    """
    x, y = _finite(y_true, y_pred)
    if x.size:
        # Limit the view to the bulk of the data so a few huge errors do not
        # squash the plot.
        if one_to_one:
            lo = float(min(np.percentile(x, 0.5), np.percentile(y, 0.5)))
            hi = float(max(np.percentile(x, 99.5), np.percentile(y, 99.5)))
            x_lo, x_hi, y_lo, y_hi = lo, hi, lo, hi
        else:
            x_lo, x_hi = (float(v) for v in np.percentile(x, [0.5, 99.5]))
            y_lo, y_hi = (float(v) for v in np.percentile(y, [0.5, 99.5]))
        # hexbin needs a non-empty range on each axis.
        if x_lo == x_hi:
            x_lo, x_hi = x_lo - 1.0, x_hi + 1.0
        if y_lo == y_hi:
            y_lo, y_hi = y_lo - 1.0, y_hi + 1.0
        hb = ax.hexbin(x, y, gridsize=60, extent=(x_lo, x_hi, y_lo, y_hi),
                       mincnt=1, bins="log")
        ax.figure.colorbar(hb, ax=ax, label="rows (log scale)")
        if one_to_one:
            ax.plot([x_lo, x_hi], [y_lo, y_hi], "k--", linewidth=1, label="1:1")
            ax.legend(loc="upper left")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


def plot_error_vs_delta(ax, delta_k, error, min_delta=None, ylabel="Error", title=None):
    """Density plot of estimation error against the outdoor/return temperature gap.

    This shows why the mixing equation is ill-conditioned: errors grow as
    ``|T_outdoor - T_return|`` approaches zero.

    Args:
        ax: Matplotlib axes to draw on.
        delta_k: ``|T_outdoor - T_return|`` in K (x axis).
        error: Estimation error (same length). Non-finite pairs are skipped.
        min_delta: If given, a vertical line marks the ill-conditioning threshold.
        ylabel: Label for the y axis (include units).
        title: Optional axes title.

    Returns:
        The axes, for chaining.
    """
    x, y = _finite(delta_k, error)
    if x.size:
        x_hi = float(np.percentile(x, 99.5))
        y_lo, y_hi = (float(v) for v in np.percentile(y, [1, 99]))
        if y_lo == y_hi:                      # flat data: widen so hexbin has area
            y_lo, y_hi = y_lo - 1.0, y_hi + 1.0
        hb = ax.hexbin(x, y, gridsize=60, extent=(0, x_hi, y_lo, y_hi), mincnt=1, bins="log")
        ax.figure.colorbar(hb, ax=ax, label="rows (log scale)")
        ax.axhline(0.0, color="k", linewidth=0.8)
        if min_delta is not None:
            ax.axvline(min_delta, color="r", linestyle="--", linewidth=1,
                       label=f"threshold {min_delta:g} K")
            ax.legend(loc="upper right")
        ax.set_xlim(0, x_hi)
        ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("|T_outdoor - T_return| (K)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax
