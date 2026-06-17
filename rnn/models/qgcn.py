"""Q-GCN / HGCN wrapper (plan section 5, Corollary 11 / prediction P3).

This wraps the **vendored, torch-2.x-ported** QGCN stack (``external/QGCN``) --
the pseudo-Riemannian Q-GCN of Xiong et al., the Corollary-11 model whose
curvature penalty is *additive* across layers
``Psi_curv = sum_l c_l sqrt(|kappa^(l)|)`` (prediction P3).

Port notes (the torch-1.1 -> 2.6 work):
- The only import-time breakage was an unused ``from torchvision import transforms``
  in the pseudo-hyperboloid manifolds (patched out in the submodule).
- The pseudo-hyperboloid ``mobius_add`` (the hyperbolic-bias path) is numerically
  unstable and trips ``assert not isnan`` on a plain forward pass, so we run with
  ``use_bias=False`` (a standard HGCN configuration). All other ops are stable in
  float32 on CPU/CUDA.

We feed the QGCN layers our own clean adjacency/feature tensors (from
:mod:`rnn.data.real`), so QGCN's rotten ``utils/data_utils`` loaders are never
imported. The vendored QGCN root is put on ``sys.path`` lazily inside
:func:`build_qgcn` so importing :mod:`rnn.models` stays light.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

_QGCN_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external",
    "QGCN",
)


def _ensure_qgcn_on_path():
    if _QGCN_ROOT not in sys.path:
        sys.path.insert(0, _QGCN_ROOT)


class QGCNNodeClassifier(nn.Module):
    """A Q-GCN / HGCN node classifier built on the ported QGCN encoder.

    Pipeline: Euclidean input projection (feat -> dim) -> QGCN hyperbolic-GCN
    encoder on the pseudo-hyperboloid (``depth`` curved layers) -> tangent-space
    linear classifier (logmap0 then Linear). Cross-entropy is the task loss; the
    validated quantity is the gap ``Delta = test_loss - train_loss`` (plan section 7).

    ``self.curvatures`` is the per-layer curvature list (the kappa^(l) of P3); the
    additive penalty ``sum_l sqrt(|kappa^(l)|)`` is :meth:`curvature_penalty`.
    """

    def __init__(
        self,
        feat_dim: int,
        n_classes: int,
        *,
        dim: int = 16,
        depth: int = 2,
        time_dim: int = 2,
        c: float = 1.0,
        manifold: str = "PseudoHyperboloid",
        act: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        _ensure_qgcn_on_path()
        from models.encoders import HGCN

        assert depth >= 2, "QGCN encoder requires num_layers > 1"
        assert time_dim < dim, "pseudo-hyperboloid needs time_dim < dim"
        self.dim = dim
        self.depth = depth
        self.c_value = float(c)

        # Euclidean input projection keeps the manifold lift well-conditioned
        # (raw high-dim sparse features lifted directly onto the pseudo-hyperboloid
        # are numerically fragile).
        self.in_proj = nn.Linear(feat_dim, dim)

        args = SimpleNamespace(
            manifold=manifold, space_dim=dim - time_dim, time_dim=time_dim,
            num_layers=depth, feat_dim=dim, dim=dim, task="nc",
            act=act, dropout=dropout, bias=0,           # bias off (see port notes)
            use_att=0, local_agg=0, c=float(c), cuda=-1, device="cpu",
        )
        self._c = torch.tensor([float(c)])
        self.encoder = HGCN(self._c, args)
        # Per-layer curvatures (kappa^(l) of P3). HGCN stores |kappa| (beta>0).
        self.curvatures = [float(t.detach().reshape(-1)[0]) for t in self.encoder.curvatures]

        self.classifier = nn.Linear(dim, n_classes)
        self.manifold = self.encoder.manifold

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        emb = self.encoder.encode(h, adj)           # points on the manifold
        tangent = self.manifold.logmap0(emb, self._c.to(emb.device))
        return self.classifier(tangent)

    def curvature_penalty(self) -> float:
        """The Cor. 11 additive curvature penalty ``sum_l sqrt(|kappa^(l)|)`` (P3).

        Additive (a sum over layers), in contrast to the multiplicative ``Lambda_N``
        of the general bound (Theorem 10). Testing that stacking curved layers
        *adds* (not multiplies) penalty terms is prediction P3.
        """
        import math

        return float(sum(math.sqrt(abs(k)) for k in self.curvatures))


def build_qgcn(feat_dim: int, n_classes: int, **kwargs) -> QGCNNodeClassifier:
    """Construct a runnable Q-GCN / HGCN node classifier (see :class:`QGCNNodeClassifier`)."""
    return QGCNNodeClassifier(feat_dim, n_classes, **kwargs)
