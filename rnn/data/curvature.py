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
import math
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
    """Negative-curvature estimate from delta and graph diameter.

    We use the standard delta-hyperbolicity -> curvature relation: a delta-hyperbolic
    geodesic space embeds with curvature bounded above by roughly ``-(log 3 / 2 delta)^2``
    (the constant-curvature hyperbolic space H_kappa has delta = log(3)/(2 sqrt(|kappa|)),
    so |kappa| ~ (log 3 / (2 delta))^2). We normalise delta by the graph radius
    (diameter / 2) to make it scale-free, matching the unit-radius synthetic tube.

    Returns a signed ``kappa`` on the same scale as the synthetic sweep so real
    points can be overlaid. This is an *estimate* (plan section 9: real graphs are
    not constant-curvature); label it as such.
    """
    if diameter <= 0:
        return 0.0
    radius = diameter / 2.0
    # Graphs have an integer shortest-path metric, so the smallest *resolvable*
    # non-zero delta is 0.5. A tree measures delta == 0 (maximally hyperbolic);
    # flooring at the 0.5 quantum keeps the curvature estimate finite ("at least
    # this hyperbolic") instead of diverging to -inf.
    delta_eff = max(delta, 0.5)
    # Scale-free hyperbolicity: delta relative to the graph radius.
    delta_rel = max(delta_eff / radius, 1e-6)
    kappa_mag = (math.log(3.0) / (2.0 * delta_rel)) ** 2
    return -kappa_mag


def dataset_curvature(name: str, num_samples: int = 50_000, seed: int = 0) -> dict:
    """Load a real dataset's graph, measure delta-hyperbolicity, estimate curvature.

    Returns ``{delta, diameter, kappa_est, sqrt_abs_kappa_est, n_nodes, n_edges}``.
    Prefers the largest connected component (diameter/shortest paths need
    connectivity). This is the placement coordinate for the phase-diagram anchors
    (plan section 3).
    """
    import networkx as nx

    from rnn.data import real

    g = real.load_graph(name)
    if g.number_of_nodes() == 0:
        raise ValueError(f"empty graph for {name!r}")
    # Largest connected component (shortest-path metrics need connectivity).
    if not nx.is_connected(g):
        cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(cc).copy()

    delta = delta_hyperbolicity_sample(g, num_samples=num_samples, seed=seed)
    diameter = float(nx.diameter(g))
    kappa = estimated_curvature_from_delta(delta, diameter)
    return {
        "dataset": name,
        "delta": float(delta),
        "diameter": diameter,
        "kappa_est": float(kappa),
        "sqrt_abs_kappa_est": math.sqrt(abs(kappa)),
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
    }
