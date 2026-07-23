"""Run the descent optimization experiment and write the paper figures.

    python run.py

Regenerates the paper's optimization figures (Section 'Experiments', fig:descent
and fig:collapse) and RESULTS.md:
  - the K->0 collapse of the intrinsic loss onto the Euclidean surrogate;
  - the near-optimum landscape sharpness lambda*_K vs signed curvature K (the
    late-phase step-size mechanism), and its scaling against S_K(R)^2;
  - the delta-balancedness defect along training.

Honest framing: the step-size ceiling eta*_K = O(1/S_K(R)^2) is a worst-case bound
whose clean witness is the late-phase sharpness. We report that sharpness and its
S_K(R)^2 scaling, not an inflated whole-trajectory eta*.

Figures are written to the repo root and copied into paper/ so \\includegraphics
resolves. Both are gitignored and fully regenerable.
"""

from __future__ import annotations

import os

import figures
from optimization import DescentConfig, run_descent_experiment

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(ROOT, "paper")


def main():
    cfg = DescentConfig()
    exp = run_descent_experiment(cfg)

    descent = figures.plot_descent(exp, os.path.join(ROOT, "figure_descent.pdf"))
    collapse = figures.plot_optimization_collapse(
        exp, os.path.join(ROOT, "figure_collapse.pdf"))
    # copy next to the paper source so \includegraphics resolves.
    if os.path.isdir(PAPER_DIR):
        for src in (descent, collapse):
            with open(src, "rb") as f, \
                 open(os.path.join(PAPER_DIR, os.path.basename(src)), "wb") as g:
                g.write(f.read())

    _write_md(exp, cfg, descent, collapse)
    _print_summary(exp)
    print(f"- figures: {os.path.basename(descent)}, {os.path.basename(collapse)} "
          f"(root + paper/)")
    print("- results: RESULTS.md")


def _print_summary(exp):
    Ks = sorted(exp["curvatures"])
    print("# Optimization experiment (paper Section 'Experiments')")
    print(f"- radius R = {exp['radius']}, seeds = {exp['n_seeds']}")
    print(f"- K=0 intrinsic == Euclidean surrogate (collapse): "
          f"exponent |b| intrinsic={exp['exponents'][0.0]:.4g} vs "
          f"surrogate={exp['exponents']['surrogate']:.4g}")
    print("- near-optimum sharpness lambda*_K, predicted step 2/lambda*, vs S_K(R)^2:")
    for K in Ks:
        print(f"    K={K:>5g}:  lambda*={exp['sharpness'][K]:.4g}"
              f"  (rel {exp['sharpness_rel'][K]:.3g})"
              f"  2/lambda*={exp['eta_pred'][K]:.4g}"
              f"  S_K(R)^2={exp['s2'][K]:.4g}")
    khyp = min(Ks); ksph = max(Ks)
    print(f"- hyperbolic sharpening K={khyp:g}: lambda*/lambda*_0 = "
          f"{exp['sharpness_rel'][khyp]:.3g} (predicted step {exp['eta_pred'][khyp]:.3g} "
          f"vs Euclidean {exp['eta_pred'][0.0]:.3g})")
    print(f"- spherical relaxation K={ksph:g}: lambda*/lambda*_0 = "
          f"{exp['sharpness_rel'][ksph]:.3g} (predicted step {exp['eta_pred'][ksph]:.3g} "
          f"> Euclidean {exp['eta_pred'][0.0]:.3g})")


def _write_md(exp, cfg, descent, collapse):
    Ks = sorted(exp["curvatures"])
    lines = [
        "# Results: curvature and the deep-linear step size", "",
        f"Deep-linear network on the kappa-Stereographic model, depth {cfg.depth}, "
        f"width {cfg.width}, m={cfg.n_samples}, radius R={cfg.radius}, "
        f"{exp['n_seeds']} seeds. Regenerate: `python run.py`.", "",
        "Signed sectional curvature K: K<0 hyperbolic, K=0 Euclidean, K>0 spherical.", "",
        "## Near-optimum sharpness vs curvature (the step-size mechanism)", "",
        "The step-size ceiling eta*_K = O(1/S_K(R)^2) is a worst-case bound; its clean "
        "witness is the late-phase landscape sharpness lambda*_K (top Hessian eigenvalue "
        "of the intrinsic loss at the converged solution), where the paper's "
        "adaptive-schedule analysis gives L -> S_K(R)^2 * mean||xi||^2. It increases "
        "with hyperbolic curvature and decreases for spherical curvature, matching "
        "S_K(R)^2 in the moderate regime and exceeding it at strong hyperbolic curvature "
        "(the H_K, B_K terms).", "",
        "| K | S_K(R)^2 | lambda*_K | lambda*_K/lambda*_0 | 2/lambda*_K (pred. step) |",
        "|---|---|---|---|---|",
    ]
    for K in Ks:
        lines.append(f"| {K:g} | {exp['s2'][K]:.3f} | {exp['sharpness'][K]:.4f} "
                     f"(±{exp['sharpness_ci'][K]:.4f}) | {exp['sharpness_rel'][K]:.3f} "
                     f"| {exp['eta_pred'][K]:.4f} |")
    lines += [
        "", f"![descent]({os.path.basename(descent)})", "",
        "## Collapse and linear convergence", "",
        "At K=0 the intrinsic loss equals the Euclidean tangent surrogate exactly; the "
        "per-step convergence exponent |b| coincides with the surrogate there and moves "
        "with curvature away from it.", "",
        f"- intrinsic |b| at K=0: {exp['exponents'][0.0]:.4g}; "
        f"surrogate |b|: {exp['exponents']['surrogate']:.4g}", "",
        f"![collapse]({os.path.basename(collapse)})", "",
    ]
    with open(os.path.join(ROOT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
