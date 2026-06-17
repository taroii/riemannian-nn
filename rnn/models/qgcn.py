"""Q-GCN / HGCN wrappers (plan section 5, Corollary 11) -- DEFERRED to the server.

The pseudo-Riemannian Q-GCN of Xiong et al. (the Corollary-11 model, with the
additive layerwise curvature penalty Psi_curv = sum_l c_l sqrt(|kappa^(l)|),
prediction P3) is vendored under ``external/QGCN``. Its manifolds, layers, and
optimizers (RiemannianAdam) live there, but the package targets torch==1.1 and
needs PyTorch-2.x patching before it imports cleanly.

That patching is server work (the laptop smoke test runs only the clean
geoopt synthetic path). This module is a placeholder so the import surface and
the plan's section-10 layout exist now; fill it in on the GPU box, reusing
``external/QGCN/models`` and ``external/QGCN/layers/hyp_layers.py`` and exposing
the same per-layer (alpha_j, beta_j, P_j) trace that :mod:`rnn.instrument` needs.
"""

from __future__ import annotations


def build_qgcn(*args, **kwargs):
    raise NotImplementedError(
        "Q-GCN wrapper is deferred to the server run. Vendored sources are in "
        "external/QGCN (models/, layers/hyp_layers.py, optimizers/radam.py); patch "
        "them for PyTorch 2.x there, then expose a per-layer trace compatible with "
        "rnn.instrument.lambda_n.instrument_lambda_n."
    )
