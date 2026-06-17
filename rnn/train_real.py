"""Train a Q-GCN / HGCN on a real dataset and return a measured-gap row.

This produces the real-data anchors for the phase diagram / collapse plot (plan
section 3). For each real dataset we:

1. load the graph + features + labels (clean loaders, :mod:`rnn.data.real`);
2. place it by curvature via sampled delta-hyperbolicity (:mod:`rnn.data.curvature`);
3. train the ported Q-GCN node classifier and measure the gap
   ``Delta = test_loss - train_loss`` (plan section 7 -- the gap, never test error).

Real anchors *corroborate* the synthetic contours; they are not load-bearing
(plan section 3). Curvature for real graphs is a delta-hyperbolicity *estimate*
(plan section 9): a real graph is not constant-curvature.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from rnn.data import curvature, real
from rnn.models.qgcn import build_qgcn


def _normalized_adj(adj: sp.csr_matrix) -> torch.Tensor:
    """Symmetric-normalized adjacency with self-loops, as a torch sparse tensor."""
    adj = adj + sp.eye(adj.shape[0])
    deg = np.asarray(adj.sum(1)).flatten()
    dinv_sqrt = np.zeros_like(deg, dtype=np.float64)
    nz = deg > 0
    dinv_sqrt[nz] = np.power(deg[nz], -0.5)
    d = sp.diags(dinv_sqrt)
    a = (d @ adj @ d).tocoo()
    idx = torch.from_numpy(np.vstack((a.row, a.col)).astype(np.int64))
    val = torch.from_numpy(a.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, a.shape).coalesce()


def run_real(
    dataset: str = "cora",
    *,
    depth: int = 2,
    dim: int = 16,
    time_dim: int = 2,
    c: float = 1.0,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 0,
    device: str = "cpu",
    delta_samples: int = 20_000,
) -> dict:
    """Train Q-GCN on a real node-classification dataset; return a flat result row.

    Runs on CPU by default: the vendored QGCN manifold code holds its curvature
    tensors off the module (not registered buffers) and has hardcoded device
    assumptions, so it does not cleanly move to CUDA. The real graphs are tiny
    (Cora ~2.7k nodes), so CPU is the appropriate target -- the GPU is reserved
    for the synthetic headline sweep.
    """
    torch.manual_seed(seed)
    bundle = real.load(dataset)
    if bundle["task"] != "nc":
        raise ValueError(
            f"{dataset!r} has no node labels (task={bundle['task']}); the real "
            "anchor currently supports node-classification datasets (e.g. cora)."
        )

    feats = bundle["features"]
    x = torch.tensor(np.asarray(feats.todense()), dtype=torch.float32, device=device)
    adj = _normalized_adj(bundle["adj"]).to(device)
    y = torch.tensor(bundle["labels"], dtype=torch.long, device=device)
    idx_tr = torch.tensor(bundle["idx_train"], dtype=torch.long, device=device)
    idx_te = torch.tensor(bundle["idx_test"], dtype=torch.long, device=device)

    model = build_qgcn(
        feat_dim=x.shape[1], n_classes=bundle["n_classes"],
        dim=dim, depth=depth, time_dim=time_dim, c=c,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[idx_tr], y[idx_tr])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x, adj)
        train_loss = float(F.cross_entropy(logits[idx_tr], y[idx_tr]).item())
        test_loss = float(F.cross_entropy(logits[idx_te], y[idx_te]).item())
        train_acc = float((logits[idx_tr].argmax(1) == y[idx_tr]).float().mean().item())
        test_acc = float((logits[idx_te].argmax(1) == y[idx_te]).float().mean().item())

    curv = curvature.dataset_curvature(dataset, num_samples=delta_samples, seed=seed)
    return {
        "dataset": dataset,
        "model": "qgcn",
        "depth": depth,
        "dim": dim,
        "time_dim": time_dim,
        "c_model": c,
        "seed": seed,
        "delta": curv["delta"],
        "diameter": curv["diameter"],
        "kappa_data": curv["kappa_est"],
        "sqrt_abs_kappa_data": curv["sqrt_abs_kappa_est"],
        "n_nodes": curv["n_nodes"],
        "train_loss": train_loss,
        "test_loss": test_loss,
        "gap": test_loss - train_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "curvature_penalty_P3": model.curvature_penalty(),
    }


def gather_anchors(
    datasets=("cora",),
    depths=(2, 3, 4),
    *,
    epochs: int = 200,
    seeds=(0,),
    out_path: str | None = None,
) -> pd.DataFrame:
    """Train Q-GCN across datasets x depths x seeds; return + cache the anchor rows.

    These are the real-data anchors overlaid on the phase diagram (plan section 3).
    Stacking depth tests the Cor. 11 additive penalty `sum_l sqrt(|kappa^l|)` (P3):
    ``curvature_penalty_P3`` grows linearly in depth at fixed per-layer curvature.
    """
    rows = []
    for ds in datasets:
        for depth in depths:
            for seed in seeds:
                rows.append(run_real(ds, depth=depth, epochs=epochs, seed=seed))
    df = pd.DataFrame(rows)
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "real_anchors.parquet",
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
