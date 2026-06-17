"""Generate all figures + the metrics summary from a results parquet (plan section 8).

    python -m analysis.run_analysis --results results/headline.parquet

Writes the phase diagram, collapse, P1-slope, and Lambda_N-collapse PDFs to
``figures/`` and prints (and writes ``results/RESULTS_summary.md``) the fitted
constant, held-out error placeholder, P1 slope/R^2, and Kendall tau.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn.analysis import figures, metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--y-axis", default="depth", choices=["depth", "kappa_model"])
    args = ap.parse_args()

    df = pd.read_parquet(args.results)
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.results))[0]

    paths = {
        "phase_diagram": figures.plot_phase_diagram(
            df, os.path.join(FIG_DIR, f"{tag}_phase.pdf"), y_axis=args.y_axis
        ),
        "collapse": figures.plot_collapse(df, os.path.join(FIG_DIR, f"{tag}_collapse.pdf")),
        "p1_slope": figures.plot_p1_slope(df, os.path.join(FIG_DIR, f"{tag}_p1.pdf")),
        "lambda_collapse": figures.plot_lambda_collapse(
            df, os.path.join(FIG_DIR, f"{tag}_lambda.pdf")
        ),
    }

    p1 = metrics.p1_slope(df)
    tau = metrics.kendall_tau(df)
    collapse = metrics.collapse_fit(df)

    lines = [
        f"# Analysis summary: {tag}",
        "",
        f"- runs: {len(df)}  |  cells: {collapse['n_cells']}",
        f"- **P1** gap ~ sqrt(|kappa_data|): slope={p1['slope']:.4g}, "
        f"R^2={p1['r2']:.3f}, p={p1['p_value']:.2g}",
        f"- **collapse** global constant C={collapse['global_constant']:.4g}, "
        f"R^2_all={collapse['r2_all']:.3f}",
        f"- **Kendall tau** (bound_full vs gap): tau={tau['tau']:.3f}, "
        f"p={tau['p_value']:.2g} over {tau['n_cells']} cells",
        "",
        "## Figures",
    ] + [f"- {k}: `{os.path.relpath(v, ROOT)}`" for k, v in paths.items()]

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(ROOT, "results", "RESULTS_summary.md"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
