"""Figure generation for the tree-embedding experiment.

Matplotlib only. Three figures, all comparing Euclidean vs Poincare-ball embeddings
of a tree:
- :func:`plot_tree_embedding`   distortion + reconstruction mAP vs embedding dim
- :func:`plot_tree_generalization`  held-out distortion vs fraction of pairs held out
- :func:`plot_shepard`          scaled embedding distance vs true graph distance
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: server-safe, no display needed
import matplotlib.pyplot as plt
import numpy as np

_TWO_COL = (3.4, 2.8)


def plot_tree_embedding(df, out_path: str) -> str:
    """Two panels: distortion and reconstruction mAP vs embedding dim, Euclid vs hyper.

    ``df`` has columns space, dim, distortion, recon_map (one row per space x dim x seed).
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    colors = {"euclid": "tab:blue", "hyper": "tab:red"}
    labels = {"euclid": "Euclidean", "hyper": "Hyperbolic"}
    for space in ["euclid", "hyper"]:
        sub = df[df["space"] == space]
        agg = sub.groupby("dim").agg(
            dist_m=("distortion", "mean"), dist_s=("distortion", "std"),
            map_m=("recon_map", "mean"), map_s=("recon_map", "std"),
        ).reset_index()
        x = agg["dim"].to_numpy()
        axes[0].errorbar(x, agg["dist_m"], yerr=agg["dist_s"], fmt="o-", ms=4, lw=1.0,
                         color=colors[space], label=labels[space])
        axes[1].errorbar(x, agg["map_m"], yerr=agg["map_s"], fmt="o-", ms=4, lw=1.0,
                         color=colors[space], label=labels[space])
    axes[0].set_xlabel("embedding dim"); axes[0].set_ylabel("distortion")
    axes[0].set_yscale("log"); axes[0].set_title("(a) distance distortion"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("embedding dim"); axes[1].set_ylabel("reconstruction mAP")
    axes[1].set_title("(b) tree reconstruction"); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tree_generalization(df, out_path: str) -> str:
    """Held-out distortion vs fraction of node-pairs held out (Euclid vs hyper).

    ``df``: space, holdout_frac, seed, train_distortion, heldout_distortion. Solid =
    held-out (generalization), dashed = train (fit).
    """
    fig, ax = plt.subplots(figsize=_TWO_COL)
    colors = {"euclid": "tab:blue", "hyper": "tab:red"}
    labels = {"euclid": "Euclidean", "hyper": "Hyperbolic"}
    for space in ["euclid", "hyper"]:
        sub = df[df["space"] == space]
        agg = sub.groupby("holdout_frac").agg(
            ho_m=("heldout_distortion", "mean"), ho_s=("heldout_distortion", "std"),
            tr_m=("train_distortion", "mean"),
        ).reset_index()
        x = agg["holdout_frac"].to_numpy()
        ax.errorbar(x, agg["ho_m"], yerr=agg["ho_s"], fmt="o-", ms=4, lw=1.1,
                    color=colors[space], label=f"{labels[space]} (held-out)")
        ax.plot(x, agg["tr_m"], "--", lw=0.8, color=colors[space], alpha=0.6)
    ax.set_xlabel("fraction of pairs held out")
    ax.set_ylabel("distortion")
    ax.set_yscale("log")
    ax.set_title("generalization: held-out (solid) vs train (dashed)", fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_shepard(euc: dict, hyp: dict, out_path: str) -> str:
    """Shepard diagram: scaled embedding distance vs true graph distance, both spaces.

    ``euc``/``hyp`` are :func:`rnn.embed.embed_graph` outputs with ``return_pairs=True``
    (carrying ``dG_pairs`` and ``pred_dist``). Hyperbolic points hug the diagonal;
    Euclidean scatter -- the fit-quality view.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for ax, res, color, title in [
        (axes[0], euc, "tab:blue", "Euclidean"),
        (axes[1], hyp, "tab:red", "Hyperbolic"),
    ]:
        dg, pred = res["dG_pairs"], res["pred_dist"]
        ax.scatter(dg, pred, s=6, alpha=0.15, color=color)
        lim = [0, float(np.max(dg)) + 1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{title}\navg distortion={res['distortion']:.3f}")
        ax.set_xlabel("graph distance"); ax.set_aspect("equal")
    axes[0].set_ylabel("embedding distance (scaled)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
