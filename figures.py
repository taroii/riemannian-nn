"""Figure generation for the optimization experiments (paper Section 'Experiments').

Matplotlib only. Every plot writes a PDF and returns the path.

  plot_descent(core)          fig:descent  -- 4 panels: collapse+convergence,
                              near-optimum sharpness vs signed K, S_K(R)^2 scaling,
                              delta-balancedness.
  plot_collapse(core)         fig:collapse -- convergence exponent |b| vs K.
  plot_scaling(scaling)       fig:scaling  -- robustness: lambda*/lambda*_0 vs
                              S_K(R)^2 collapses onto y=x across radii + architectures.
  plot_landscape(landscape)   fig:landscape -- final losses of many balanced inits
                              all below the global-min threshold (no spurious minima).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_TWO_COL = (3.4, 2.8)


def plot_descent(core: dict, out_path: str) -> str:
    Ks = sorted(core["curvatures"])
    R = core["radius"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0))
    cmap = plt.get_cmap("viridis")
    order = {K: i for i, K in enumerate(Ks)}

    # (a) raw-loss trajectories; K=0 coincides with the Euclidean surrogate.
    ax = axes[0]
    for K in Ks:
        ax.semilogy(core["trajectories"][K], color=cmap(order[K] / max(len(Ks) - 1, 1)),
                    lw=1.0, label=f"K={K:g}")
    ax.semilogy([max(v, 1e-16) for v in core["surrogate_traj"]], "k--", lw=1.2,
                label="surrogate E")
    ax.set_xlabel("GD step"); ax.set_ylabel(r"intrinsic loss $L$")
    ax.set_title("(a) linear convergence; K→0 collapse"); ax.legend(fontsize=5, ncol=2)

    # (b) near-optimum sharpness vs signed curvature, with CIs.
    ax = axes[1]
    lam = [core["sharpness"][K] for K in Ks]
    ci = [core["sharpness_ci"][K] for K in Ks]
    ax.errorbar(Ks, lam, yerr=ci, fmt="o-", lw=1.0, ms=4, color="C3", capsize=2)
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("signed curvature K"); ax.set_ylabel(r"sharpness $\lambda^\star_K$")
    ax.set_title("(b) landscape sharpness at optimum")
    ax.annotate("spherical\n(step relaxes)", xy=(0.02, 0.92), xycoords="axes fraction",
                fontsize=5, ha="left", va="top")
    ax.annotate("hyperbolic\n(step shrinks)", xy=(0.98, 0.92), xycoords="axes fraction",
                fontsize=5, ha="right", va="top")

    # (c) scaling test: lambda*_K/lambda*_0 vs S_K(R)^2, identity line, CI bars.
    ax = axes[2]
    x = np.array([core["s2"][K] for K in Ks])
    y = np.array([core["sharpness_rel"][K] for K in Ks])
    yerr = np.array([core["sharpness_rel_ci"][K] for K in Ks])
    lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="identity")
    ax.errorbar(x, y, yerr=yerr, fmt="o", ms=4, lw=0.7, capsize=2,
                color="C0", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S_K(R)^2$ (theory)")
    ax.set_ylabel(r"$\lambda^\star_K/\lambda^\star_0$ (measured)")
    ax.set_title(f"(c) scaling test, R={R:g}"); ax.legend(fontsize=6)

    # (d) balancedness defect along training.
    ax = axes[3]
    for K in Ks:
        ax.semilogy(core["defect_traj"][K], color=cmap(order[K] / max(len(Ks) - 1, 1)),
                    lw=1.0, label=f"K={K:g}")
    ax.set_xlabel("GD step")
    ax.set_ylabel(r"$\max_j\|W_{j+1}^\top W_{j+1}-W_jW_j^\top\|_F$")
    ax.set_title("(d) δ-balancedness"); ax.legend(fontsize=5, ncol=2)

    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_collapse(core: dict, out_path: str) -> str:
    Ks = sorted(core["curvatures"])
    fig, ax = plt.subplots(figsize=_TWO_COL)
    ax.plot(Ks, [core["exponents"][K] for K in Ks], "o-", lw=1.0, ms=4,
            label=r"intrinsic $|b|$")
    ax.axhline(core["exponents"]["surrogate"], ls="--", color="k", lw=1.0,
               label="surrogate (K=0)")
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("signed curvature K"); ax.set_ylabel(r"convergence exponent $|b|$")
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_scaling(scaling: dict, out_path: str) -> str:
    """Robustness collapse: every (R, arch, seed, K) point on the identity line."""
    pts = scaling["points"]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    radii = sorted({p["R"] for p in pts})
    archs = sorted({(p["depth"], p["width"]) for p in pts})
    cmap = plt.get_cmap("plasma")
    markers = ["o", "s", "^", "D", "v", "P"]
    rcol = {R: cmap(i / max(len(radii) - 1, 1)) for i, R in enumerate(radii)}
    amark = {a: markers[i % len(markers)] for i, a in enumerate(archs)}
    for p in pts:
        ax.scatter(p["s2"], p["rel"], s=14, alpha=0.5,
                   color=rcol[p["R"]], marker=amark[(p["depth"], p["width"])])
    allx = [p["s2"] for p in pts] + [p["rel"] for p in pts]
    lo, hi = min(allx) * 0.7, max(allx) * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="identity")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S_K(R)^2$ (theory)")
    ax.set_ylabel(r"$\lambda^\star_K/\lambda^\star_0$ (measured)")
    ax.set_title("Curvature scaling collapses across R and architecture")
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], marker="o", ls="", color=rcol[R], label=f"R={R:g}") for R in radii]
    h2 = [Line2D([], [], marker=amark[a], ls="", color="gray",
                 label=f"N={a[0]},d={a[1]}") for a in archs]
    leg1 = ax.legend(handles=h1, fontsize=6, loc="upper left", title="radius")
    ax.add_artist(leg1)
    ax.legend(handles=h2 + [Line2D([], [], ls="--", color="k", label="identity")],
              fontsize=6, loc="lower right", title="architecture")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_landscape(landscape: dict, out_path: str) -> str:
    """Final losses of many balanced inits, per curvature; all below threshold."""
    Ks = sorted(landscape.keys())
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=_TWO_COL)
    for i, K in enumerate(Ks):
        f = np.asarray(landscape[K]["finals"]).clip(1e-30, None)
        ax.scatter(np.full_like(f, i) + rng.uniform(-0.12, 0.12, size=f.shape),
                   f, s=12, alpha=0.6)
    ax.axhline(1e-3, ls="--", color="k", lw=1.0, label="global-min threshold")
    ax.set_yscale("log"); ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"K={K:g}" for K in Ks])
    ax.set_ylabel("final loss (random balanced inits)")
    ax.set_title("No spurious minima on the tube"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path
