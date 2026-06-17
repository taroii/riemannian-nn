"""Figure generation (plan section 8). All plots write a PDF and return the path.

Matplotlib only (no seaborn). These are bespoke because the figures ARE the
contribution (plan section 0). The phase diagram is the headline (plan section 3);
collapse / slope / Lambda_N-collapse go to the appendix (plan section 4).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: server-safe, no display needed
import matplotlib.pyplot as plt
import numpy as np

from rnn.analysis import metrics

# AAAI two-column width is ~3.3in; full width ~7in.
_TWO_COL = (3.4, 2.8)
_FULL = (7.0, 3.0)


def plot_phase_diagram(df, out_path: str, y_axis: str = "depth") -> str:
    """Headline: heatmap of the measured gap over sqrt(|kappa_data|) x model-knob.

    Predicted iso-gap contours (global constant fitted on all cells here; swap to
    a held-out fit for the decisiveness version of plan section 3) are overlaid.
    """
    cells = metrics.cell_means(df)
    xs = np.sort(cells["sqrt_abs_kappa_data"].unique())
    ys = np.sort(cells[y_axis].unique())
    gap_grid = np.full((len(ys), len(xs)), np.nan)
    pred_grid = np.full((len(ys), len(xs)), np.nan)
    C = metrics.fit_global_constant(
        cells["bound_predictor"].to_numpy(), cells["gap_mean"].to_numpy()
    )
    for _, r in cells.iterrows():
        i = int(np.searchsorted(ys, r[y_axis]))
        j = int(np.searchsorted(xs, r["sqrt_abs_kappa_data"]))
        gap_grid[i, j] = r["gap_mean"]
        pred_grid[i, j] = C * r["bound_predictor"]

    fig, ax = plt.subplots(figsize=_TWO_COL)
    im = ax.imshow(
        gap_grid, origin="lower", aspect="auto", cmap="magma",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    )
    # Predicted iso-gap contours overlaid on the measured heatmap.
    if np.isfinite(pred_grid).sum() >= 4:
        X, Y = np.meshgrid(xs, ys)
        try:
            cs = ax.contour(X, Y, pred_grid, colors="cyan", linewidths=0.8, alpha=0.9)
            ax.clabel(cs, inline=True, fontsize=5)
        except Exception:
            pass
    ax.set_xlabel(r"$\sqrt{|\kappa_{\mathrm{data}}|}$")
    ax.set_ylabel({"depth": r"depth $N$", "kappa_model": r"$\kappa_{\mathrm{model}}$"}.get(y_axis, y_axis))
    fig.colorbar(im, ax=ax, label=r"gap $\Delta$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_collapse(df, out_path: str) -> str:
    """Appendix: measured gap vs the scalar bound predictor; should collapse to a line."""
    cells = metrics.cell_means(df)
    fit = metrics.collapse_fit(df)
    C = fit["global_constant"]
    x = cells["bound_predictor"].to_numpy()
    y = cells["gap_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=_TWO_COL)
    ax.errorbar(x, y, yerr=cells["gap_ci95"].to_numpy(), fmt="o", ms=3, lw=0.6, alpha=0.8)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, C * xx, "k--", lw=1.0, label=fr"$C\cdot$pred, $R^2$={fit['r2_all']:.2f}")
    ax.set_xlabel("bound predictor (eq. 16 numerator / rate)")
    ax.set_ylabel(r"measured gap $\Delta$")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_p1_slope(df, out_path: str, fixed_depth: int | None = None) -> str:
    """Appendix P1 panel: gap vs sqrt(|kappa_data|) with the linear fit."""
    fit = metrics.p1_slope(df, fixed_depth=fixed_depth)
    d = df if fixed_depth is None else df[df["depth"] == fixed_depth]
    x = d["sqrt_abs_kappa_data"].to_numpy()
    y = d["gap"].to_numpy()

    fig, ax = plt.subplots(figsize=_TWO_COL)
    ax.scatter(x, y, s=8, alpha=0.5)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, fit["slope"] * xx + fit["intercept"], "r-", lw=1.0,
            label=fr"slope={fit['slope']:.3g}, $R^2$={fit['r2']:.2f}")
    ax.set_xlabel(r"$\sqrt{|\kappa_{\mathrm{data}}|}$")
    ax.set_ylabel(r"gap $\Delta$")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lambda_collapse(df, out_path: str) -> str:
    """Appendix P2 panel: gap vs sqrt(d0 * log Lambda_N) across depth/curvature/width."""
    cells = metrics.cell_means(df)
    d0 = float(df["d0"].iloc[0])
    x = np.sqrt(d0 * np.log(np.clip(cells["Lambda_N"].to_numpy(), 1.0 + 1e-9, None)))
    y = cells["gap_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=_TWO_COL)
    ax.errorbar(x, y, yerr=cells["gap_ci95"].to_numpy(), fmt="o", ms=3, lw=0.6, alpha=0.8)
    ax.set_xlabel(r"$\sqrt{d_0 \log \Lambda_N}$")
    ax.set_ylabel(r"measured gap $\Delta$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
