"""Run the optimization experiments and write all paper figures + RESULTS.md.

    python run.py            # QUICK: local sanity pass (few seeds, one R/arch)
    python run.py --full     # FULL: server sweep (many seeds, R x architecture grid)

Experiments (all honest, path A):
  E1  K->0 collapse + linear convergence            fig:descent (a)
  E2  near-optimum sharpness ~ S_K(R)^2, sign flip  fig:descent (b,c), fig:scaling
  E3  delta-balancedness maintained                 fig:descent (d)
  E7  no spurious minima on the tube                fig:landscape
  GC  gradient correctness (autograd vs finite diff) RESULTS.md

Figures land in the repo root and are copied into paper/ so \\includegraphics
resolves. Raw sweep points are written to scaling_points.csv for re-analysis
without recompute. Everything is gitignored and regenerable.
"""

from __future__ import annotations

import argparse
import os

import figures
import optimization as O

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(ROOT, "paper")


def quick_config():
    return O.Config(
        n_seeds=5, train_steps=4000,
        sweep_radii=(1.2,), sweep_arch=((3, 6),),
        sweep_curvatures=(0.5, 0.0, -0.5, -1.0, -2.0, -4.0),
        landscape_inits=12,
    )


def full_config():
    return O.Config(
        n_seeds=50, train_steps=8000,
        curvatures=(1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -4.0, -6.0, -8.0),
        sweep_radii=(0.8, 1.0, 1.2, 1.5),
        sweep_arch=((2, 4), (3, 6), (4, 10)),
        sweep_curvatures=(0.5, 0.0, -0.5, -1.0, -2.0, -4.0, -6.0),
        landscape_inits=50,
    )


def _copy_to_paper(path):
    if os.path.isdir(PAPER_DIR):
        with open(path, "rb") as f, open(os.path.join(PAPER_DIR, os.path.basename(path)), "wb") as g:
            g.write(f.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="server sweep (many seeds/grid)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel worker processes (0 = auto: all cores-2 in --full)")
    args = ap.parse_args()
    cfg = full_config() if args.full else quick_config()
    if args.jobs > 0:
        cfg.jobs = args.jobs
    elif args.full:
        cfg.jobs = max(1, (os.cpu_count() or 2) - 2)
    print(f"[run] mode={'FULL' if args.full else 'QUICK'}  seeds={cfg.n_seeds}  "
          f"radii={cfg.sweep_radii}  arch={cfg.sweep_arch}  jobs={cfg.jobs}")

    core = O.run_core(cfg)
    scaling = O.run_scaling(cfg)
    landscape = O.run_landscape(cfg)
    gcheck = O.gradient_check(cfg)

    paths = {
        "figure_descent.pdf": figures.plot_descent(core, os.path.join(ROOT, "figure_descent.pdf")),
        "figure_collapse.pdf": figures.plot_collapse(core, os.path.join(ROOT, "figure_collapse.pdf")),
        "figure_scaling.pdf": figures.plot_scaling(scaling, os.path.join(ROOT, "figure_scaling.pdf")),
        "figure_landscape.pdf": figures.plot_landscape(landscape, os.path.join(ROOT, "figure_landscape.pdf")),
    }
    for p in paths.values():
        _copy_to_paper(p)

    _write_csv(scaling)
    _write_md(core, scaling, landscape, gcheck, cfg)
    _print_summary(core, scaling, landscape, gcheck)
    print("- figures:", ", ".join(paths))
    print("- results: RESULTS.md ; raw sweep: scaling_points.csv")


def _collapse_stats(scaling):
    """Geometric-mean ratio measured/predicted and fraction within 2x -> robustness."""
    import numpy as np
    r = np.array([p["rel"] / p["s2"] for p in scaling["points"] if p["s2"] > 0])
    return float(np.exp(np.mean(np.log(r)))), float(((r > 0.5) & (r < 2.0)).mean()), len(r)


def _print_summary(core, scaling, landscape, gcheck):
    Ks = sorted(core["curvatures"])
    gm, frac2x, n = _collapse_stats(scaling)
    print("# Optimization experiments")
    print(f"- E1 collapse: |b| intrinsic(K=0)={core['exponents'][0.0]:.4g} == "
          f"surrogate={core['exponents']['surrogate']:.4g}")
    print(f"- E2 sharpness rel (measured/S^2): geo-mean={gm:.3f}, "
          f"{frac2x:.0%} within 2x over {n} sweep points")
    for K in Ks:
        print(f"    K={K:>5g}: lambda*={core['sharpness'][K]:.4g} "
              f"rel={core['sharpness_rel'][K]:.3g} S^2={core['s2'][K]:.4g}")
    print("- E7 landscape (frac reaching global | max final loss):")
    for K in sorted(landscape):
        print(f"    K={K:>5g}: {landscape[K]['frac_global']:.0%} | "
              f"{landscape[K]['max_final']:.2e}")
    print("- GC gradient max-rel-error:",
          ", ".join(f"K={K:g}:{e:.1e}" for K, e in gcheck.items()))


def _write_csv(scaling):
    import csv
    with open(os.path.join(ROOT, "scaling_points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["R", "depth", "width", "seed", "K", "s2", "rel"])
        w.writeheader()
        for p in scaling["points"]:
            w.writerow(p)


def _write_md(core, scaling, landscape, gcheck, cfg):
    Ks = sorted(core["curvatures"])
    gm, frac2x, n = _collapse_stats(scaling)
    L = [
        "# Results: curvature and the deep-linear step size", "",
        f"kappa-Stereographic deep-linear net, depth {cfg.depth}, width {cfg.width}, "
        f"m={cfg.n_samples}, {core['n_seeds']} seeds. Signed curvature K (<0 hyperbolic, "
        ">0 spherical). Regenerate: `python run.py` (quick) / `python run.py --full` (server).",
        "",
        "## E1 collapse + linear convergence  (Thm convergence, Prop surrogate)",
        f"- intrinsic |b| at K=0 = {core['exponents'][0.0]:.5g}; "
        f"surrogate |b| = {core['exponents']['surrogate']:.5g}  (identical => exact collapse)",
        "", f"![descent](figure_descent.pdf)", "",
        "## E2 near-optimum sharpness ~ S_K(R)^2  (step-size mechanism, Cor positive)", "",
        "| K | S_K(R)^2 | lambda*_K | lambda*_K/lambda*_0 |",
        "|---|---|---|---|",
    ]
    for K in Ks:
        L.append(f"| {K:g} | {core['s2'][K]:.3f} | {core['sharpness'][K]:.4f} "
                 f"(±{core['sharpness_ci'][K]:.4f}) | {core['sharpness_rel'][K]:.3f} "
                 f"(±{core['sharpness_rel_ci'][K]:.3f}) |")
    L += [
        "",
        f"Robustness (fig:scaling): across radii {list(cfg.sweep_radii)} and "
        f"architectures {[list(a) for a in cfg.sweep_arch]}, the ratio measured/predicted "
        f"has geometric mean {gm:.3f} and {frac2x:.0%} of {n} points fall within 2x of "
        "the S_K(R)^2 prediction.",
        "", f"![scaling](figure_scaling.pdf)", "",
        "## E7 no spurious minima on the tube  (Thm landscape)", "",
        "| K | frac reaching global (loss<1e-3) | max final loss |",
        "|---|---|---|",
    ]
    for K in sorted(landscape):
        L.append(f"| {K:g} | {landscape[K]['frac_global']:.0%} | {landscape[K]['max_final']:.2e} |")
    L += [
        "", f"![landscape](figure_landscape.pdf)", "",
        "## GC gradient correctness (autograd vs central differences)",
        "- max relative error: " + ", ".join(f"K={K:g}: {e:.1e}" for K, e in gcheck.items()),
        "",
    ]
    with open(os.path.join(ROOT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
