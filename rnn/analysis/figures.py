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


def _holdout_mask_corner(cells, y_axis="depth", frac=0.35):
    """Hold out the high-curvature x high-depth corner (the decisiveness check).

    Fit the global constant on the rest, predict this corner: plan section 3's
    'fit on a subset, predict held-out curvatures and depths'.
    """
    x = cells["sqrt_abs_kappa_data"].to_numpy()
    yv = cells[y_axis].to_numpy()
    x_thr = np.quantile(np.unique(x), 1 - frac)
    y_thr = np.quantile(np.unique(yv), 1 - frac)
    return (x >= x_thr) & (yv >= y_thr)


def plot_phase_diagram(df, out_path: str, y_axis: str = "depth", real_anchors=None) -> str:
    """Headline: heatmap of the measured gap over sqrt(|kappa_data|) x model-knob.

    Predicted iso-gap contours use a constant fitted on the grid MINUS the
    high-curvature/high-depth corner, then extrapolated over the whole grid (the
    decisiveness check of plan section 3). Real datasets, if given, are overlaid as
    labelled position markers at their measured delta-curvature (their gap is in a
    different loss unit, so they mark placement, not color -- see RESULTS.md).
    """
    cells = metrics.cell_means(df)
    xs = np.sort(cells["sqrt_abs_kappa_data"].unique())
    ys = np.sort(cells[y_axis].unique())
    gap_grid = np.full((len(ys), len(xs)), np.nan)
    pred_grid = np.full((len(ys), len(xs)), np.nan)
    holdout = _holdout_mask_corner(cells, y_axis)
    train = ~holdout
    C = metrics.fit_global_constant(
        cells["bound_predictor"].to_numpy()[train], cells["gap_mean"].to_numpy()[train]
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
    # Real-data anchors: labelled position markers at measured delta-curvature.
    if real_anchors is not None and len(real_anchors):
        ra = real_anchors
        rx = ra["sqrt_abs_kappa_data"].to_numpy()
        ry = ra[y_axis].to_numpy() if y_axis in ra else ra["depth"].to_numpy()
        ax.scatter(rx, ry, marker="x", s=40, c="white", linewidths=1.4, zorder=5)
        for xi, yi, name in zip(rx, ry, ra["dataset"]):
            ax.annotate(str(name), (xi, yi), fontsize=5, color="white",
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(r"$\sqrt{|\kappa_{\mathrm{data}}|}$")
    ax.set_ylabel({"depth": r"depth $N$", "kappa_model": r"$\kappa_{\mathrm{model}}$"}.get(y_axis, y_axis))
    fig.colorbar(im, ax=ax, label=r"gap $\Delta$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_phase_diagram_3d(df, out_path: str, y_axis: str = "depth", real_anchors=None) -> str:
    """Headline, as an intuitive 3D landscape (alternative to the heatmap+contours).

    The measured gap is a solid surface over (sqrt|kappa_data|, depth); height = gap,
    so the 'failure corner' is literally a peak. The theory's predicted bound
    (global constant fitted off the held-out corner) is drawn as a red wireframe
    floating over it -- if the bound captures the *shape*, the wireframe tracks the
    surface. Real datasets, if given, are vertical stems at their delta-curvature.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    from matplotlib.lines import Line2D

    cells = metrics.cell_means(df)
    xs = np.sort(cells["sqrt_abs_kappa_data"].unique())
    ys = np.sort(cells[y_axis].unique())
    X, Y = np.meshgrid(xs, ys)
    Zg = np.full(X.shape, np.nan)
    Zp = np.full(X.shape, np.nan)
    holdout = _holdout_mask_corner(cells, y_axis)
    train = ~holdout
    C = metrics.fit_global_constant(
        cells["bound_predictor"].to_numpy()[train], cells["gap_mean"].to_numpy()[train]
    )
    for _, r in cells.iterrows():
        i = int(np.searchsorted(ys, r[y_axis]))
        j = int(np.searchsorted(xs, r["sqrt_abs_kappa_data"]))
        Zg[i, j] = r["gap_mean"]
        Zp[i, j] = C * r["bound_predictor"]

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Zg, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True,
        rstride=1, cstride=1,
    )
    ax.plot_wireframe(X, Y, Zp, color="crimson", linewidth=0.7, alpha=0.8)
    # Set z to the measured-surface range so the synthetic shape stays readable.
    zlo, zhi = float(np.nanmin(Zg)), float(np.nanmax([np.nanmax(Zg), np.nanmax(Zp)]))
    ax.set_zlim(zlo, zhi)
    if real_anchors is not None and len(real_anchors):
        # Real gaps are a different loss unit (CE vs synthetic MSE), so they would
        # wreck the z-scale -- place them as FLOOR markers at their delta-curvature
        # (their actual contribution is *curvature placement*, not a comparable gap).
        ra = real_anchors
        rx = ra["sqrt_abs_kappa_data"].to_numpy()
        ry = ra[y_axis].to_numpy() if y_axis in ra else ra["depth"].to_numpy()
        ax.scatter(rx, ry, zlo, color="black", marker="^", s=28, zorder=6)
        for xi, yi, name in zip(rx, ry, ra["dataset"]):
            ax.text(xi, yi, zlo, f" {name}", fontsize=6)

    ax.set_xlabel(r"$\sqrt{|\kappa_{\mathrm{data}}|}$", labelpad=6)
    ax.set_ylabel({"depth": "depth $N$", "kappa_model": r"$\kappa_{\mathrm{model}}$"}.get(y_axis, y_axis), labelpad=6)
    ax.set_zlabel(r"gap $\Delta$", labelpad=6)
    ax.view_init(elev=26, azim=-58)
    ax.set_title("measured gap (surface) vs predicted bound shape (red mesh)", fontsize=9)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label=r"gap $\Delta$")
    legend = [
        Line2D([0], [0], color="crimson", lw=1.0, label=r"predicted $C\cdot$bound"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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


def plot_descent(exp: dict, out_path: str) -> str:
    """Four-panel optimization figure (paper fig:descent) from run_descent_experiment."""
    curvs = exp["curvatures"]
    R = exp["radius"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 2.9))

    # (a) loss-excess trajectories; surrogate dashed; c=0 should coincide with it.
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(curvs):
        ax.semilogy(exp["trajectories"][c], color=cmap(i / max(len(curvs) - 1, 1)),
                    lw=1.0, label=f"c={c:g}")
    surr = exp["surrogate_traj"]
    floor = min(surr)
    ax.semilogy([max(v - floor, 1e-16) for v in surr], "k--", lw=1.2, label="surrogate E")
    ax.set_xlabel("GD step"); ax.set_ylabel("loss excess")
    ax.set_title("(a) trajectories; c→0 collapse"); ax.legend(fontsize=5, ncol=2)

    # (b) max stable step size vs curvature.
    ax = axes[1]
    ax.semilogy(curvs, [exp["eta_star"][c] for c in curvs], "o-", lw=1.0, ms=4)
    ax.set_xlabel("curvature c"); ax.set_ylabel(r"$\eta^\star$")
    ax.set_title("(b) max stable step")

    # (c) normalized eta* vs 1/S_c(R)^2.
    ax = axes[2]
    ax.plot(curvs, [exp["eta_norm"][c] for c in curvs], "o-", lw=1.0, ms=4, label=r"$\eta^\star/\eta^\star_0$")
    ax.plot(curvs, [exp["inv_s2"][c] for c in curvs], "s--", lw=1.0, ms=4, label=r"$1/S_c(R)^2$")
    ax.set_xlabel("curvature c"); ax.set_ylabel("normalized")
    ax.set_title(f"(c) vs $1/S_c(R)^2$, R={R:g}"); ax.legend(fontsize=6); ax.set_yscale("log")

    # (d) balancedness defect along training.
    ax = axes[3]
    for i, c in enumerate(curvs):
        ax.plot(exp["defect_traj"][c], color=cmap(i / max(len(curvs) - 1, 1)), lw=1.0, label=f"c={c:g}")
    ax.set_xlabel("GD step"); ax.set_ylabel("balancedness defect")
    ax.set_title("(d) δ-balancedness"); ax.legend(fontsize=5, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_optimization_collapse(exp: dict, out_path: str) -> str:
    """Collapse check (paper fig:collapse): convergence exponent |b| vs curvature."""
    curvs = exp["curvatures"]
    fig, ax = plt.subplots(figsize=_TWO_COL)
    ax.plot(curvs, [exp["exponents"][c] for c in curvs], "o-", lw=1.0, ms=4, label=r"intrinsic $|b|$")
    ax.axhline(exp["exponents"]["surrogate"], ls="--", color="k", lw=1.0, label="surrogate (c=0)")
    ax.set_xlabel("curvature c"); ax.set_ylabel(r"convergence exponent $|b|$")
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
