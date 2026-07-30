"""Figure generation for the optimization experiments (paper §4.8; Appendix B).

Matplotlib only. Every plot writes a PDF and returns the path.

Design: figures are authored at their display size (so LaTeX does not scale text
down), fonts are large, and explanatory text lives in the CAPTIONS, not the axes.
Many-curvature panels use a colorbar for K instead of a crowded legend.

  plot_descent(core)          fig:descent  -- 4 panels: collapse+convergence,
                              near-optimum sharpness vs signed K, S_K(R)^2 scaling,
                              delta-balancedness.
  plot_collapse(core)         fig:collapse -- single-step exponent |b| vs K.
  plot_scaling(scaling)       fig:scaling  -- lambda*/lambda*_0 vs S_K(R)^2 across
                              R, architecture, seed (in-regime vs boundary).
  plot_landscape(landscape)   fig:landscape -- final losses of many balanced inits.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Author at display size with readable fonts; captions carry the prose.
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "lines.linewidth": 1.6,
    "axes.grid": False,
})

_CMAP = plt.get_cmap("viridis")


def _kcolorbar(fig, ax, Ks, label="curvature $K$"):
    norm = mpl.colors.Normalize(vmin=min(Ks), vmax=max(Ks))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=_CMAP)
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=9)
    return norm


def plot_descent(core: dict, out_path: str) -> str:
    Ks = sorted(core["curvatures"])
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4))

    # (a) raw-loss trajectories, colored by K; K=0 coincides with surrogate.
    ax = axes[0]
    norm = mpl.colors.Normalize(vmin=min(Ks), vmax=max(Ks))
    for K in Ks:
        ax.semilogy(core["trajectories"][K], color=_CMAP(norm(K)), lw=1.4)
    ax.semilogy([max(v, 1e-16) for v in core["surrogate_traj"]], "--", color="crimson",
                lw=2.0, label="surrogate $\\mathcal{E}$")
    ax.set_xlabel("GD step"); ax.set_ylabel(r"intrinsic loss $L$")
    ax.set_title(r"(a) convergence \& $K\!\to\!0$ collapse" if mpl.rcParams["text.usetex"]
                 else "(a) convergence and $K\\!\\to\\!0$ collapse")
    ax.legend(loc="upper right", frameon=False)

    # (b) near-optimum sharpness vs signed curvature, with CIs.
    ax = axes[1]
    lam = [core["sharpness"][K] for K in Ks]
    ci = [core["sharpness_ci"][K] for K in Ks]
    ax.errorbar(Ks, lam, yerr=ci, fmt="o-", ms=6, color="C3", capsize=3)
    ax.axvline(0.0, color="0.5", lw=1.0, ls=":")
    ax.set_xlabel(r"signed curvature $K$"); ax.set_ylabel(r"sharpness $\lambda^\star_K$")
    ax.set_title("(b) sharpness at optimum")

    # (c) scaling test: lambda*_K/lambda*_0 vs S_K(R)^2, identity line, CI bars.
    ax = axes[2]
    x = np.array([core["s2"][K] for K in Ks])
    y = np.array([core["sharpness_rel"][K] for K in Ks])
    yerr = np.array([core["sharpness_rel_ci"][K] for K in Ks])
    lo, hi = min(x.min(), y.min()) * 0.7, max(x.max(), y.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1.5, label="identity")
    ax.errorbar(x, y, yerr=yerr, fmt="o", ms=6, capsize=3, color="C0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S_K(R)^2$"); ax.set_ylabel(r"$\lambda^\star_K/\lambda^\star_0$")
    ax.set_title("(c) scaling vs $S_K(R)^2$")
    ax.legend(loc="upper left", frameon=False)

    # (d) balancedness defect along training, colored by K.
    ax = axes[3]
    for K in Ks:
        ax.semilogy(core["defect_traj"][K], color=_CMAP(norm(K)), lw=1.4)
    ax.set_xlabel("GD step"); ax.set_ylabel(r"balancedness defect")
    ax.set_title(r"(d) $\delta$-balancedness")
    _kcolorbar(fig, ax, Ks)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_collapse(core: dict, out_path: str) -> str:
    Ks = sorted(core["curvatures"])
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ax.plot(Ks, [core["exponents"][K] for K in Ks], "o-", ms=6, label=r"intrinsic")
    ax.axhline(core["exponents"]["surrogate"], ls="--", color="crimson", lw=1.8,
               label="surrogate ($K{=}0$)")
    ax.axvline(0.0, color="0.5", lw=1.0, ls=":")
    ax.set_xlabel(r"signed curvature $K$")
    ax.set_ylabel(r"convergence exponent $|b|$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scaling(scaling: dict, out_path: str, regime=2.0) -> str:
    """lambda*_K/lambda*_0 vs S_K(R)^2. In-regime (sqrt|K|R<=regime) colored by R;
    boundary points greyed. The collapse statistic goes in the caption."""
    pts = [p for p in scaling["points"]
           if p["s2"] > 0 and math.isfinite(p["rel"]) and p["rel"] > 0]
    def t(p):
        return math.sqrt(abs(p["K"])) * p["R"]
    inr = [p for p in pts if t(p) <= regime]
    out = [p for p in pts if t(p) > regime]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    radii = sorted({p["R"] for p in pts})
    archs = sorted({(p["depth"], p["width"]) for p in pts})
    markers = ["o", "s", "^", "D", "v", "P"]
    amark = {a: markers[i % len(markers)] for i, a in enumerate(archs)}
    rnorm = mpl.colors.Normalize(vmin=min(radii), vmax=max(radii))
    rcmap = plt.get_cmap("viridis")

    for p in out:                      # boundary regime: greyed background
        ax.scatter(p["s2"], p["rel"], s=14, alpha=0.15, color="0.6",
                   marker=amark[(p["depth"], p["width"])], zorder=1, linewidths=0)
    for p in inr:                      # controlled regime: colored by R
        ax.scatter(p["s2"], p["rel"], s=20, alpha=0.6, color=rcmap(rnorm(p["R"])),
                   marker=amark[(p["depth"], p["width"])], zorder=2, linewidths=0)

    allx = [p["s2"] for p in pts] + [p["rel"] for p in pts]
    lo, hi = min(allx) * 0.7, max(allx) * 1.4
    ax.plot([lo, hi], [lo, hi], "--", color="k", lw=1.8, zorder=3, label="identity")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S_K(R)^2$ (theory)")
    ax.set_ylabel(r"$\lambda^\star_K/\lambda^\star_0$ (measured)")

    sm = mpl.cm.ScalarMappable(norm=rnorm, cmap=rcmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"radius $R$ (in-regime)"); cb.ax.tick_params(labelsize=9)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=amark[a], ls="", color="gray",
                      label=f"$N{{=}}{a[0]},d{{=}}{a[1]}$") for a in archs]
    handles += [Line2D([], [], marker="o", ls="", color="0.6",
                       label=r"$\sqrt{|K|}\,R>2$"),
                Line2D([], [], ls="--", color="k", label="identity")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_surrogate(sg: dict, out_path: str) -> str:
    """Surrogate is curvature-free (Proposition 12). (a) intrinsic loss under
    intrinsic GD separates with K; (b) surrogate loss under surrogate GD is
    identical across K (curves coincide). Mean over seeds."""
    Ks = sorted(sg["curvatures"])
    norm = mpl.colors.Normalize(vmin=min(Ks), vmax=max(Ks))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), sharey=True)
    panels = [(axes[0], "intrinsic", r"$L_K(t)/L_K(0)$", "(a) intrinsic loss, intrinsic GD"),
              (axes[1], "surrogate", r"$\mathcal{E}(t)/\mathcal{E}(0)$",
               "(b) surrogate loss, surrogate GD")]
    for ax, key, ylab, ttl in panels:
        for K in Ks:
            m = np.maximum(np.asarray(sg[key][K]["mean"]), 1e-16)
            ax.semilogy(np.arange(len(m)), m, color=_CMAP(norm(K)), lw=1.8)
        ax.set_xlabel("GD step"); ax.set_title(ttl)
        ax.set_ylabel(ylab)
    _kcolorbar(fig, axes[1], Ks)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_landscape(landscape: dict, out_path: str) -> str:
    Ks = sorted(landscape.keys())
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    for i, K in enumerate(Ks):
        f = np.asarray(landscape[K]["finals"]).clip(1e-30, None)
        ax.scatter(np.full_like(f, i) + rng.uniform(-0.13, 0.13, size=f.shape),
                   f, s=22, alpha=0.6, color="C0", linewidths=0)
    ax.axhline(1e-3, ls="--", color="k", lw=1.6)
    ax.set_yscale("log")
    ax.set_ylim(top=1e-2)   # headroom so the threshold line is not flush with the top spine
    # the 10^{-3} global-min threshold is explained in the caption (keeps the axes uncluttered)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"${K:g}$" for K in Ks])
    ax.set_xlabel(r"signed curvature $K$")
    ax.set_ylabel("final loss")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
