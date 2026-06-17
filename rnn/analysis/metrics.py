"""Quantitative analysis of the results table (plan section 8).

All functions take the per-(config, seed) DataFrame produced by the sweep and
return plain numbers / arrays. They operate on aggregated (mean-over-seed) cells
where appropriate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cell_means(df: pd.DataFrame, by=("kappa_data", "kappa_model", "depth", "width")) -> pd.DataFrame:
    """Mean gap (+/- CI) per grid cell, collapsing seeds."""
    by = [c for c in by if c in df.columns]
    grouped = df.groupby(by)
    out = grouped.agg(
        gap_mean=("gap", "mean"),
        gap_std=("gap", "std"),
        n_seeds=("gap", "count"),
        Lambda_N=("Lambda_N", "mean"),
        bound_predictor=("bound_predictor", "mean"),
        bound_full=("bound_full", "mean"),
        sqrt_abs_kappa_data=("sqrt_abs_kappa_data", "mean"),
    ).reset_index()
    out["gap_ci95"] = 1.96 * out["gap_std"] / np.sqrt(out["n_seeds"].clip(lower=1))
    return out


def p1_slope(df: pd.DataFrame, fixed_depth: int | None = None) -> dict:
    """P1: linear fit of the gap vs sqrt(|kappa_data|) at fixed architecture.

    Prediction P1 is that the curvature-attributable gap scales as
    ``sqrt(|kappa_data|)`` (square root, not linear in kappa). We regress the
    measured gap on ``sqrt(|kappa_data|)`` and report slope and R^2.
    """
    d = df if fixed_depth is None else df[df["depth"] == fixed_depth]
    x = d["sqrt_abs_kappa_data"].to_numpy()
    y = d["gap"].to_numpy()
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue ** 2),
        "p_value": float(res.pvalue),
        "n": int(len(x)),
    }


def kendall_tau(df: pd.DataFrame, bound_col: str = "bound_full") -> dict:
    """Appendix rank-correlation: Kendall tau of the computed bound vs the gap.

    The "Fantastic Generalization Measures" check (plan sections 0, 6): does the
    bound correctly *rank* configurations by generalization, even though its
    magnitude is loose?
    """
    cells = cell_means(df)
    tau, p = stats.kendalltau(cells[bound_col], cells["gap_mean"])
    return {"tau": float(tau), "p_value": float(p), "n_cells": int(len(cells))}


def fit_global_constant(predictor: np.ndarray, gap: np.ndarray) -> float:
    """Single global multiplicative constant C minimizing ||gap - C predictor||^2."""
    predictor = np.asarray(predictor, float)
    gap = np.asarray(gap, float)
    denom = float((predictor * predictor).sum())
    return float((predictor * gap).sum() / denom) if denom > 0 else 0.0


def collapse_fit(df: pd.DataFrame, holdout_mask: np.ndarray | None = None) -> dict:
    """Collapse plot fit (plan section 4): gap vs the scalar bound predictor.

    Fits the global constant on the (optionally held-out-excluded) data and reports
    R^2 on all points plus, if a holdout is given, the held-out extrapolation error
    -- the decisiveness check of plan section 3.
    """
    cells = cell_means(df)
    predictor = cells["bound_predictor"].to_numpy()
    gap = cells["gap_mean"].to_numpy()

    if holdout_mask is None:
        train = np.ones(len(cells), dtype=bool)
    else:
        train = ~np.asarray(holdout_mask, dtype=bool)

    C = fit_global_constant(predictor[train], gap[train])
    pred_gap = C * predictor
    ss_res = float(((gap - pred_gap) ** 2).sum())
    ss_tot = float(((gap - gap.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    out = {"global_constant": C, "r2_all": r2, "n_cells": int(len(cells))}
    if holdout_mask is not None and (~train).any():
        ho = ~train
        rel_err = np.abs(gap[ho] - pred_gap[ho]) / np.abs(gap[ho]).clip(1e-12)
        out["heldout_rel_error_mean"] = float(rel_err.mean())
        out["heldout_n"] = int(ho.sum())
    return out
