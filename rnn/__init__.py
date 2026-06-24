"""Hyperbolic vs Euclidean graph embedding (experiments for the Riemannian NN paper).

The experiment shows the architecture's value proposition: embedding hierarchical
(tree) structure into a Poincare ball achieves far lower distortion and better
generalization than a Euclidean embedding at the same dimension -- the
representational-capacity advantage the theory is built on.

Modules:
- :mod:`rnn.manifolds`   curvature conventions + geoopt Stereographic construction
- :mod:`rnn.data`        tree / hierarchy structure generators
- :mod:`rnn.embed`       Euclidean vs Poincare-ball embedding + distortion / mAP metrics
- :mod:`rnn.models`      the Euclidean MLP baseline
- :mod:`rnn.analysis`    figures (distortion/mAP vs dim, generalization, Shepard)

Driver: ``analysis/run_tree_embedding.py``.
"""

__all__ = ["manifolds", "data", "models", "analysis", "embed"]
