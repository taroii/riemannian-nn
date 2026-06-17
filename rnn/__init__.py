"""Curvature-adaptive generalization for layerwise Riemannian neural networks.

This package implements the experimental pipeline described in ``notes/plan.md``
to validate the *shape* of the generalization bound of Theorem 10 /
Corollary 11 in ``paper/main.tex``.

Subpackages mirror the plan's repo layout (plan section 10):

- :mod:`rnn.data`        synthetic generators (known kappa) + real loaders + delta-hyperbolicity
- :mod:`rnn.models`      layerwise Riemannian NN, hyperbolic MLP, (later) HGCN / Q-GCN wrappers
- :mod:`rnn.instrument`  Lambda_N estimation (alpha_j, beta_j, P_j via power iteration) + bound eval
- :mod:`rnn.analysis`    contour fit, collapse, P1 slope, Kendall tau

The driver scripts live in ``sweeps/`` (grid runner) and ``analysis/`` (figures),
both of which import from this package.
"""

__all__ = ["manifolds", "data", "models", "instrument", "analysis", "train"]
