"""Run the descent optimization experiment and write the appendix figures.

    python -m analysis.run_optimization

Regenerates the paper's optimization figures (Section 'Experiments',
fig:descent and fig:collapse) with the current code: the c->0 collapse, the
maximum stable step size eta* vs curvature, eta* vs 1/S_c(R)^2, and the
delta-balancedness defect along training.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn.analysis import figures  # noqa: E402
from rnn.optimization import DescentConfig, run_descent_experiment  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    exp = run_descent_experiment(DescentConfig())

    descent = figures.plot_descent(exp, os.path.join(FIG_DIR, "figure_descent.pdf"))
    collapse = figures.plot_optimization_collapse(
        exp, os.path.join(FIG_DIR, "figure_collapse.pdf")
    )

    print("# Optimization experiment (paper Section 'Experiments')")
    print(f"- radius R = {exp['radius']}")
    print("- eta* (max stable step) by curvature:")
    for c in exp["curvatures"]:
        print(f"    c={c:>4g}:  eta*={exp['eta_star'][c]:.4g}  "
              f"eta*/eta0={exp['eta_norm'][c]:.4g}  1/S_c(R)^2={exp['inv_s2'][c]:.4g}")
    e0, e8 = exp["eta_star"][exp["curvatures"][0]], exp["eta_star"][exp["curvatures"][-1]]
    if e8 > 0:
        print(f"- eta* drops by factor ~{e0 / e8:.1f} from c={exp['curvatures'][0]:g} "
              f"to c={exp['curvatures'][-1]:g}")
    print(f"- figures: {os.path.relpath(descent, ROOT)}, {os.path.relpath(collapse, ROOT)}")


if __name__ == "__main__":
    main()
