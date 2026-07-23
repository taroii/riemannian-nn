"""Figure generation for the optimization experiments (paper Section 'Experiments').

Matplotlib only (no seaborn). The figures ARE the contribution, so they are
bespoke. Every plot writes a PDF and returns the path.

All panels report only quantities that hold up honestly (path A):
  (a) the K->0 collapse to the Euclidean surrogate + linear convergence;
  (b) the near-optimum landscape sharpness lambda*_K vs SIGNED curvature K -- the
      late-phase step-size mechanism (hyperbolic sharpens, spherical flattens);
  (c) the scaling test lambda*_K / lambda*_0 vs S_K(R)^2 -- matches in the moderate
      regime, exceeds it at strong hyperbolic curvature (the H_K, B_K terms);
  (d) delta-balancedness stays small along training.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: server-safe, no display needed
import matplotlib.pyplot as plt
import numpy as np

# AAAI two-column width is ~3.3in; full width ~7in.
_TWO_COL = (3.4, 2.8)


def plot_descent(exp: dict, out_path: str) -> str:
    """Four-panel optimization figure (paper fig:descent) from run_descent_experiment."""
    Ks = exp["curvatures"]
    R = exp["radius"]
    # order curvatures ascending for the K-axis panels (spherical -> hyperbolic).
    Ks_sorted = sorted(Ks)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0))
    cmap = plt.get_cmap("viridis")
    order = {K: i for i, K in enumerate(Ks_sorted)}

    # (a) raw-loss trajectories; surrogate dashed; K=0 should coincide with it.
    ax = axes[0]
    for K in Ks_sorted:
        ax.semilogy(exp["trajectories"][K], color=cmap(order[K] / max(len(Ks) - 1, 1)),
                    lw=1.0, label=f"K={K:g}")
    surr = [max(v, 1e-16) for v in exp["surrogate_traj"]]
    ax.semilogy(surr, "k--", lw=1.2, label="surrogate E")
    ax.set_xlabel("GD step"); ax.set_ylabel(r"intrinsic loss $L$")
    ax.set_title("(a) linear convergence; K→0 collapse"); ax.legend(fontsize=5, ncol=2)

    # (b) near-optimum sharpness lambda*_K vs signed curvature K, with CIs.
    ax = axes[1]
    lam = [exp["sharpness"][K] for K in Ks_sorted]
    ci = [exp["sharpness_ci"][K] for K in Ks_sorted]
    ax.errorbar(Ks_sorted, lam, yerr=ci, fmt="o-", lw=1.0, ms=4, color="C3", capsize=2)
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("signed curvature K"); ax.set_ylabel(r"sharpness $\lambda^\star_K$")
    ax.set_title("(b) landscape sharpness at optimum")
    ax.annotate("spherical\n(step relaxes)", xy=(0.02, 0.9), xycoords="axes fraction",
                fontsize=5, ha="left", va="top")
    ax.annotate("hyperbolic\n(step shrinks)", xy=(0.98, 0.9), xycoords="axes fraction",
                fontsize=5, ha="right", va="top")

    # (c) scaling test: lambda*_K / lambda*_0 vs S_K(R)^2 (identity line).
    ax = axes[2]
    x = np.array([exp["s2"][K] for K in Ks_sorted])
    y = np.array([exp["sharpness_rel"][K] for K in Ks_sorted])
    lo = min(x.min(), y.min()) * 0.8
    hi = max(x.max(), y.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="identity")
    ax.scatter(x, y, c=[cmap(order[K] / max(len(Ks) - 1, 1)) for K in Ks_sorted],
               s=28, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S_K(R)^2$ (theory)"); ax.set_ylabel(r"$\lambda^\star_K/\lambda^\star_0$ (measured)")
    ax.set_title(f"(c) scaling test, R={R:g}"); ax.legend(fontsize=6)

    # (d) balancedness defect along training.
    ax = axes[3]
    for K in Ks_sorted:
        ax.semilogy(exp["defect_traj"][K], color=cmap(order[K] / max(len(Ks) - 1, 1)),
                    lw=1.0, label=f"K={K:g}")
    ax.set_xlabel("GD step")
    ax.set_ylabel(r"$\max_j\|W_{j+1}^\top W_{j+1}-W_jW_j^\top\|_F$")
    ax.set_title("(d) δ-balancedness"); ax.legend(fontsize=5, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_optimization_collapse(exp: dict, out_path: str) -> str:
    """Collapse check (paper fig:collapse): convergence exponent |b| vs signed curvature."""
    Ks = sorted(exp["curvatures"])
    fig, ax = plt.subplots(figsize=_TWO_COL)
    ax.plot(Ks, [exp["exponents"][K] for K in Ks], "o-", lw=1.0, ms=4, label=r"intrinsic $|b|$")
    ax.axhline(exp["exponents"]["surrogate"], ls="--", color="k", lw=1.0,
               label="surrogate (K=0)")
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("signed curvature K"); ax.set_ylabel(r"convergence exponent $|b|$")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
