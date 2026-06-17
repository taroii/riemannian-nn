"""The general layerwise Riemannian neural network (paper eq. (phi-layer)).

Each layer maps a manifold point ``h_{j-1} in M_{j-1}`` to

    h_j = exp_{o_j}( F_j( log_{o_{j-1}}(h_{j-1}) ) ) in M_j,

with ``F_j`` a Euclidean tangent-space operation (here ``Linear`` + activation).
This is exactly the object the descent lemma and generalization bound are stated
for, and the hyperbolic MLP of plan section 5 is this model with Stereographic
hidden manifolds.

Curvature knobs exposed per the plan:
- ``kappa_data``  : curvature of the INPUT manifold M_0 (carried by ``beta_0``).
- ``kappa_model`` : curvature of every HIDDEN manifold (carried by ``alpha_j``,
  ``beta_j`` -- the ``Lambda_N`` channel, prediction P2).
- ``depth`` N, ``width``.

The final layer outputs to the tangent space (a Euclidean regression head, the
"tangent-space surrogate" of the paper -- it recovers the Euclidean rate
verbatim), so ``alpha_N = 1`` in the instrumentation. Switch to an intrinsic
squared-distance output later if you want the curved-output-loss variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rnn import manifolds


@dataclass
class ModelConfig:
    d0: int = 8
    width: int = 32
    depth: int = 3                # N; MUST be >= 2 (see synthetic.py caveat)
    d_out: int = 1
    kappa_data: float = -1.0      # input-manifold curvature
    kappa_model: float = -1.0     # hidden-manifold curvature
    activation: str = "tanh"      # 1-Lipschitz; keeps P_j = ||W_j|| clean
    dtype: torch.dtype = manifolds.DEFAULT_DTYPE


_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "identity": nn.Identity}


class LayerwiseRiemannianNN(nn.Module):
    """Layerwise Riemannian NN with a Euclidean tangent head.

    Manifolds:
        M_0           = Stereographic(kappa_data)        (input)
        M_1 .. M_{N-1}= Stereographic(kappa_model)       (hidden)
        output        = tangent space (Euclidean)        (regression head)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.depth >= 2, "depth N must be >= 2 (single matched-curvature layer is curvature-trivial)"
        self.cfg = cfg
        self.m_in = manifolds.stereographic(cfg.kappa_data)
        self.m_hidden = manifolds.stereographic(cfg.kappa_model)

        # Tangent operations F_1..F_N. Dimensions: d0 -> width -> ... -> d_out.
        dims = [cfg.d0] + [cfg.width] * (cfg.depth - 1) + [cfg.d_out]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[j], dims[j + 1]).to(cfg.dtype) for j in range(cfg.depth)]
        )
        act_cls = _ACT[cfg.activation]
        # Activation applied after every tangent map except the final (linear) head.
        self.acts = nn.ModuleList(
            [act_cls() for _ in range(cfg.depth - 1)] + [nn.Identity()]
        )

    def tangent_op(self, j: int, u: torch.Tensor) -> torch.Tensor:
        """F_j applied to a tangent vector (Linear then activation)."""
        return self.acts[j](self.linears[j](u))

    def forward(self, x: torch.Tensor, *, return_trace: bool = False):
        """Forward pass. ``x`` is a point on M_0 (shape ``[B, d0]``).

        If ``return_trace`` is set, also return the per-layer tensors needed by the
        Lambda_N instrumentation: the manifold point ``h`` entering each layer and
        the pre-exp tangent vector ``a`` leaving each tangent op.
        """
        N = self.cfg.depth
        h = x
        trace = {"h_in": [], "a": [], "manifold_in": [], "manifold_out": []}
        for j in range(N):
            m_prev = self.m_in if j == 0 else self.m_hidden
            u = m_prev.logmap0(h)              # beta_{j}  (j-th log; beta_0 carries kappa_data)
            a = self.tangent_op(j, u)          # P_{j+1}
            if return_trace:
                trace["h_in"].append(h)
                trace["a"].append(a)
                trace["manifold_in"].append(m_prev)
                trace["manifold_out"].append(None if j == N - 1 else self.m_hidden)
            if j < N - 1:
                h = self.m_hidden.expmap0(a)   # alpha_{j+1}
            else:
                out = a                        # final tangent (regression) output
        if return_trace:
            return out, trace
        return out
