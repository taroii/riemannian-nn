"""Naive Euclidean MLP baseline.

Treats the manifold points as ordinary vectors in ambient coordinates -- the
geometry-blind comparison point for the Riemannian NN (see
``rnn.data.hyperbolic_task``). Matched depth/width so the only difference is
geometry-awareness.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rnn import manifolds


class EuclideanMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, *, width: int = 32, depth: int = 3,
                 activation: str = "relu", dtype: torch.dtype = manifolds.DEFAULT_DTYPE):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        dims = [d_in] + [width] * (depth - 1) + [d_out]
        layers = []
        for j in range(len(dims) - 1):
            layers.append(nn.Linear(dims[j], dims[j + 1]).to(dtype))
            if j < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
