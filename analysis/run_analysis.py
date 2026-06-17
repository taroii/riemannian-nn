"""Generate all figures + the metrics summary from a results parquet (plan section 8).

    python -m analysis.run_analysis --results results/headline.parquet

Writes the phase diagram (with held-out-fit contours and any real-data anchors),
collapse, P1-slope, and Lambda_N-collapse PDFs to ``figures/``, and writes
``results/RESULTS.md`` (embedded figures + fitted constant, held-out extrapolation
error, P1 slope/R^2, Kendall tau). If ``results/real_anchors.parquet`` exists
(produced by ``rnn.train_real.gather_anchors``) the real datasets are overlaid.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn.analysis import figures, metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--y-axis", default="depth", choices=["depth", "kappa_model"])
    args = ap.parse_args()

    df = pd.read_parquet(args.results)
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.results))[0]

    # Optional real-data anchors.
    anchors = None
    anchor_path = os.path.join(RES_DIR, "real_anchors.parquet")
    if os.path.exists(anchor_path):
        anchors = pd.read_parquet(anchor_path)

    paths = {}
    # Phase diagram + P1 panel are kappa_data views: only meaningful if kappa_data
    # is actually swept (a P2 sweep fixes it -> skip, the Lambda_N collapse is the view).
    kappa_swept = df["sqrt_abs_kappa_data"].nunique() >= 2
    if kappa_swept:
        paths["phase_diagram"] = figures.plot_phase_diagram(
            df, os.path.join(FIG_DIR, f"{tag}_phase.pdf"), y_axis=args.y_axis,
            real_anchors=anchors,
        )
        paths["phase_diagram_3d"] = figures.plot_phase_diagram_3d(
            df, os.path.join(FIG_DIR, f"{tag}_phase_3d.png"), y_axis=args.y_axis,
            real_anchors=anchors,
        )
        paths["p1_slope"] = figures.plot_p1_slope(df, os.path.join(FIG_DIR, f"{tag}_p1.pdf"))
    paths["collapse"] = figures.plot_collapse(df, os.path.join(FIG_DIR, f"{tag}_collapse.pdf"))
    paths["lambda_collapse"] = figures.plot_lambda_collapse(
        df, os.path.join(FIG_DIR, f"{tag}_lambda.pdf")
    )

    p1 = metrics.p1_slope(df)
    tau = metrics.kendall_tau(df)
    collapse_all = metrics.collapse_fit(df)

    # Held-out extrapolation (decisiveness check, plan section 3): fit on the grid
    # minus the high-curvature/high-depth corner, predict that corner.
    cells = metrics.cell_means(df)
    holdout = figures._holdout_mask_corner(cells, args.y_axis)
    collapse_ho = metrics.collapse_fit(df, holdout_mask=holdout)

    _write_results_md(tag, df, paths, p1, tau, collapse_all, collapse_ho, anchors, args.y_axis)
    # Keep the short machine summary too.
    print(f"[analysis] {tag}: {len(df)} runs, {collapse_all['n_cells']} cells")
    print(f"[analysis] P1 slope={p1['slope']:.4g} R^2={p1['r2']:.3f}; "
          f"collapse R^2={collapse_all['r2_all']:.3f}; tau={tau['tau']:.3f}")
    if "heldout_rel_error_mean" in collapse_ho:
        print(f"[analysis] held-out ({collapse_ho['heldout_n']} cells) "
              f"rel-error={collapse_ho['heldout_rel_error_mean']:.3g}")
    print(f"[analysis] wrote {os.path.join('results', f'RESULTS_{tag}.md')}")


def _write_results_md(tag, df, paths, p1, tau, collapse_all, collapse_ho, anchors, y_axis):
    gap = df["gap"].to_numpy()
    lines = [
        f"# Results: {tag}",
        "",
        f"Generated from `results/{tag}.parquet` ({len(df)} runs, "
        f"{collapse_all['n_cells']} grid cells). Regenerate with "
        f"`python -m analysis.run_analysis --results results/{tag}.parquet`.",
        "",
        "## Headline numbers",
        "",
        f"- **Measured gap** Δ: mean={gap.mean():.4g}, std={gap.std():.4g}, "
        f"range=[{gap.min():.4g}, {gap.max():.4g}]",
        f"- **P1** (gap ~ √|κ_data| at fixed arch): slope={p1['slope']:.4g}, "
        f"R²={p1['r2']:.3f}, p={p1['p_value']:.2g}",
        f"- **Collapse** (gap vs bound predictor, all cells): "
        f"global constant C={collapse_all['global_constant']:.4g}, R²={collapse_all['r2_all']:.3f}",
        f"- **Kendall τ** (full bound vs gap): τ={tau['tau']:.3f}, "
        f"p={tau['p_value']:.2g} over {tau['n_cells']} cells",
    ]
    if "heldout_rel_error_mean" in collapse_ho:
        lines.append(
            f"- **Held-out extrapolation** (fit off the high-κ×high-{y_axis} corner, "
            f"predict it): {collapse_ho['heldout_n']} held-out cells, "
            f"mean rel-error={collapse_ho['heldout_rel_error_mean']:.3g}"
        )
    lines += ["", "## Figures", ""]
    for k, v in paths.items():
        rel = os.path.relpath(v, ROOT).replace(os.sep, "/")
        lines.append(f"### {k}")
        lines.append(f"![{k}]({os.path.relpath(v, RES_DIR).replace(os.sep, '/')})")
        lines.append("")

    if anchors is not None and len(anchors):
        lines += ["## Real-data anchors", "",
                  "Placed on the phase diagram at their measured δ-hyperbolicity "
                  "curvature. NOTE: real gaps are cross-entropy (node classification); "
                  "synthetic gaps are MSE (regression) — the anchors mark *curvature "
                  "placement*, not directly comparable color. P3 penalty = "
                  "Σ_l √|κ^l| (additive, Cor. 11).", "",
                  "| dataset | depth | √|κ_est| | gap (CE) | test_acc | P3 penalty |",
                  "|---|---|---|---|---|---|"]
        for _, r in anchors.iterrows():
            lines.append(
                f"| {r['dataset']} | {int(r['depth'])} | {r['sqrt_abs_kappa_data']:.3g} | "
                f"{r['gap']:.4g} | {r.get('test_acc', float('nan')):.3f} | "
                f"{r['curvature_penalty_P3']:.3g} |"
            )
        lines.append("")

    # Tag-specific so multiple sweeps (headline, headline_matched, ...) coexist.
    with open(os.path.join(RES_DIR, f"RESULTS_{tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
