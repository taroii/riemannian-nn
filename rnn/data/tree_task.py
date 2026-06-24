"""Hierarchical (tree) structure for the embedding experiment.

The canonical regime where hyperbolic geometry wins (Sarkar; Nickel & Kiela):
trees grow exponentially, so they embed with low distortion in hyperbolic space at
small dimension, while Euclidean space needs many more dimensions. Unlike the
prototype-classification task (which hands the model the coordinates), here the
input is the *graph structure* -- the model must BUILD a low-dim representation,
which is exactly the representational bottleneck hyperbolic geometry addresses.
"""

from __future__ import annotations

import networkx as nx
import torch


def balanced_tree(r: int = 2, h: int = 6):
    """Balanced ``r``-ary tree of height ``h``. Returns ``(dG, adj, meta)``.

    ``dG`` is the [n, n] all-pairs shortest-path (graph) distance tensor; ``adj`` is
    ``{node: set(neighbors)}``.
    """
    g = nx.balanced_tree(r, h)
    return _from_graph(g, name=f"balanced_tree_r{r}_h{h}")


def random_tree(n: int = 200, seed: int = 0):
    """Uniform random labelled tree on ``n`` nodes."""
    g = nx.random_labeled_tree(n, seed=seed) if hasattr(nx, "random_labeled_tree") \
        else nx.random_tree(n, seed=seed)
    return _from_graph(g, name=f"random_tree_n{n}")


def _from_graph(g, name: str):
    n = g.number_of_nodes()
    dG = torch.tensor(nx.floyd_warshall_numpy(g))
    adj = {i: set(g.neighbors(i)) for i in g.nodes()}
    meta = {
        "name": name, "n_nodes": n, "n_edges": g.number_of_edges(),
        "diameter": int(dG.max().item()),
    }
    return dG, adj, meta
