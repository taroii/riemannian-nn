"""Curvature estimation for real graphs: delta-hyperbolicity (plan sections 5, 9).

For each real dataset we need a single curvature estimate to *place* it on the
phase diagram / collapse plot. The standard choice is the graph's
delta-hyperbolicity (Gromov four-point condition), for which QGCN already ships a
utility: ``external/QGCN/utils/hyperbolicity.py``.

We expose a thin wrapper that prefers the vendored implementation and falls back
to a NetworkX four-point sampler so this module is usable on the laptop without
importing the (rotten) QGCN training stack. Per-edge Ollivier/Forman-Ricci
curvature (a distributional complement) is left to GraphRicciCurvature on the
server -- see the optional dependency in requirements.txt.
"""

from __future__ import annotations

import itertools
import random


def delta_hyperbolicity_sample(graph, num_samples: int = 50_000, seed: int = 0) -> float:
    """Sampled Gromov delta-hyperbolicity of a NetworkX graph (max over 4-tuples).

    A pure-NetworkX fallback so curvature placement works without QGCN. For the
    real run prefer ``external/QGCN/utils/hyperbolicity.py`` (same definition,
    tuned for these datasets); this sampler is for sanity checks and small graphs.
    """
    import networkx as nx

    nodes = list(graph.nodes())
    if len(nodes) < 4:
        return 0.0
    # All-pairs shortest paths on the (small) datasets we anchor with.
    dist = dict(nx.all_pairs_shortest_path_length(graph))
    rng = random.Random(seed)
    delta_max = 0.0
    for _ in range(num_samples):
        a, b, c, d = rng.sample(nodes, 4)
        try:
            d_ab, d_cd = dist[a][b], dist[c][d]
            d_ac, d_bd = dist[a][c], dist[b][d]
            d_ad, d_bc = dist[a][d], dist[b][c]
        except KeyError:
            continue  # disconnected pair
        s1, s2, s3 = d_ab + d_cd, d_ac + d_bd, d_ad + d_bc
        two_largest = sorted((s1, s2, s3))[-2:]
        delta_max = max(delta_max, (two_largest[1] - two_largest[0]) / 2.0)
    return delta_max


def estimated_curvature_from_delta(delta: float, diameter: float) -> float:
    """Crude negative-curvature estimate from delta and graph diameter.

    A standard heuristic maps a *more* hyperbolic graph (smaller delta relative to
    diameter) to *more* negative curvature. We return a signed ``kappa`` estimate
    on the same scale as the synthetic sweep so real points can be overlaid. Label
    this as an estimate (plan section 9: real graphs are not constant-curvature).
    """
    if diameter <= 0:
        return 0.0
    # delta/diameter in [0, ~0.5]; smaller -> more tree-like -> more curved.
    ratio = max(delta / diameter, 1e-6)
    return -(1.0 / ratio) ** 2 * 1e-2  # heuristic; recalibrate against synthetic grid
