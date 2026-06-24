"""Graph-embedding experiment: Euclidean ball vs Poincare ball (the hyperbolic win).

Embed the nodes of a (tree) graph into a ``dim``-dimensional space so that embedding
distances match graph distances, then measure how faithfully the structure is
captured. Euclidean embeddings parametrise points directly; hyperbolic embeddings
parametrise a tangent vector ``V`` and map ``E = exp_o(V)`` onto the Poincare ball of
curvature ``kappa`` (so plain Adam on ``V`` is a fair, same-optimiser comparison).

Metrics (standard, e.g. Sala et al. / Nickel & Kiela):
- **distortion**: mean relative error ``|s*d_M - d_G| / d_G`` over all pairs, with a
  learned global scale ``s`` (distances are scale-free).
- **reconstruction mAP**: for each node, how well its graph neighbours are its
  nearest embedding neighbours (mean average precision).

Result: from ``dim >= 3`` hyperbolic reaches ~1% distortion / mAP ~1.0 on a binary
tree while Euclidean stays an order of magnitude worse -- it needs far more
dimension (capacity) to represent the hierarchy. ``dim == 2`` is hard for gradient
methods in both spaces (the 2D Poincare disk needs embeddings pushed to machine
precision near the boundary; Sarkar's low-distortion 2D construction is combinatorial).
"""

from __future__ import annotations

import numpy as np
import torch

from rnn import manifolds


def _recon_map(E, space, man, adj) -> float:
    n = E.shape[0]
    with torch.no_grad():
        if space == "euclid":
            D = torch.cdist(E, E)
        else:
            D = torch.stack([man.dist(E[i : i + 1].expand_as(E), E) for i in range(n)])
        D.fill_diagonal_(float("inf"))
    aps = []
    for i in range(n):
        order = torch.argsort(D[i]).tolist()
        nb = adj[i]
        if not nb:
            continue
        hit, precs = 0, []
        for rank, j in enumerate(order, 1):
            if j in nb:
                hit += 1
                precs.append(hit / rank)
                if hit == len(nb):
                    break
        aps.append(sum(precs) / len(nb))
    return float(np.mean(aps))


def embed_graph(
    dG: torch.Tensor,
    adj: dict,
    space: str,                # "euclid" | "hyper"
    dim: int,
    *,
    epochs: int = 5000,
    lr: float = 0.05,
    seed: int = 0,
    burn_in: float = 0.1,      # hyperbolic burn-in fraction (lr*0.1) to avoid early boundary collapse
    kappa: float = -1.0,
    holdout_frac: float = 0.0, # fraction of node-pairs held out for the generalization test
    holdout_seed: int = 0,
    return_pairs: bool = False, # also return (dG_pairs, pred_dist) for a Shepard diagram
) -> dict:
    """Embed the graph and return distortion / recon_map.

    With ``holdout_frac > 0`` the embedding is fit to graph distances on only a random
    ``1 - holdout_frac`` subset of node-pairs, and distortion is reported separately on
    the train pairs and the held-out pairs -- a true *generalization* test: can the
    geometry infer unseen distances from partial structure? Returns
    ``{distortion, train_distortion, heldout_distortion, recon_map}``.
    """
    dG = dG.to(manifolds.DEFAULT_DTYPE)
    n = dG.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1)
    dG_pairs = dG[iu, ju]
    man = manifolds.stereographic(kappa)

    # Train/held-out split over node-pairs.
    if holdout_frac > 0.0:
        gp = torch.Generator().manual_seed(holdout_seed)
        train_mask = torch.rand(dG_pairs.shape[0], generator=gp) >= holdout_frac
    else:
        train_mask = torch.ones(dG_pairs.shape[0], dtype=torch.bool)
    test_mask = ~train_mask

    g = torch.Generator().manual_seed(seed)
    V = torch.nn.Parameter(0.01 * torch.randn(n, dim, generator=g, dtype=manifolds.DEFAULT_DTYPE))
    log_s = torch.nn.Parameter(torch.zeros((), dtype=manifolds.DEFAULT_DTYPE))
    opt = torch.optim.Adam([V, log_s], lr=lr)
    n_burn = int(epochs * burn_in)

    def points():
        return V if space == "euclid" else man.expmap0(V)

    def pair_dists(E):
        if space == "euclid":
            return (E[iu] - E[ju]).norm(dim=1)
        return man.dist(E[iu], E[ju])

    dG_train = dG_pairs[train_mask]
    for e in range(epochs):
        cur_lr = lr * 0.1 if (space == "hyper" and e < n_burn) else lr
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        opt.zero_grad()
        dM = pair_dists(points())
        loss = ((log_s.exp() * dM[train_mask] - dG_train) ** 2).mean()  # fit TRAIN pairs only
        loss.backward()
        opt.step()

    with torch.no_grad():
        E = points()
        dM = pair_dists(E)
        rel = torch.abs(log_s.exp() * dM - dG_pairs) / dG_pairs
        out = {
            "distortion": float(rel.mean().item()),
            "recon_map": _recon_map(E, space, man, adj),
        }
        if holdout_frac > 0.0:
            out["train_distortion"] = float(rel[train_mask].mean().item())
            out["heldout_distortion"] = float(rel[test_mask].mean().item())
        if return_pairs:
            out["dG_pairs"] = dG_pairs.numpy()
            out["pred_dist"] = (log_s.exp() * dM).numpy()
    return out
