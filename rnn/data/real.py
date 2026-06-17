"""Real-dataset loaders (plan section 5).

The datasets are vendored inside the QGCN submodule (``external/QGCN/data``).
QGCN's own loaders target torch==1.1 / networkx==2.2 (``from_scipy_sparse_matrix``,
``node[u]``, ``torch.sparse.FloatTensor``) and do not import on the torch-2.x /
networkx-3.x stack. Per plan section 0 we therefore **reuse the vendored data
files but parse them with fresh networkx-3 / scipy code** (the plan expects the
old loaders to be replaced, and prefers the maintained stack).

Two entry points:

- :func:`load_graph` -- structure only (a ``networkx.Graph``). This is all the
  delta-hyperbolicity placement of plan section 3 needs, and it works for every
  vendored dataset without pulling in features/labels.
- :func:`load` -- the full ``{adj, features, labels, ...}`` bundle used to train a
  real-data model for the phase-diagram anchors (plan section 3).

Two on-disk formats are present:
- ``cora`` -- the Planetoid ``ind.cora.*`` citation format (features + labels +
  the public train/val/test split).
- everything else (``disease_md``, ``grqc``, ``facebook``, ...) -- a plain
  ``<name>.edges.csv`` integer edge list (structure only; identity features).
"""

from __future__ import annotations

import os
import pickle as pkl
import sys

import numpy as np
import scipy.sparse as sp

# Datasets vendored inside the QGCN submodule.
QGCN_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "external",
    "QGCN",
    "data",
)

# Real graph datasets to anchor on the phase diagram (plan section 3), mapped to
# their folder name under ``external/QGCN/data``. Curvature placement uses
# delta-hyperbolicity (see :mod:`rnn.data.curvature`).
REAL_DATASETS = {
    "cora": "cora",            # Planetoid citation graph (features + labels)
    "disease": "disease_md",   # disease-spread tree-like graph (edge list)
    "grqc": "grqc",            # GR-QC collaboration network (edge list)
    "facebook": "facebook",    # facebook ego-network (edge list)
    "bio_diseasome": "bio-diseasome",
    "web_edu": "web-edu",
    "power": "power",
    "cs_phd": "cs_phd",
}


def available() -> dict[str, str]:
    """Return ``{name: abs_path}`` for datasets physically present in the submodule."""
    out = {}
    for name, folder in REAL_DATASETS.items():
        path = os.path.join(QGCN_DATA_ROOT, folder)
        if os.path.isdir(path):
            out[name] = path
    return out


def _folder(name: str) -> str:
    if name not in REAL_DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REAL_DATASETS)}")
    path = os.path.join(QGCN_DATA_ROOT, REAL_DATASETS[name])
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"dataset {name!r} not found at {path}. Did you run "
            "`git submodule update --init`?"
        )
    return path


# --------------------------------------------------------------------------- #
# Edge-list datasets (disease_md, grqc, facebook, ...): `<name>.edges.csv`.
# --------------------------------------------------------------------------- #
def _load_edge_list(folder: str) -> sp.csr_matrix:
    """Parse a ``<name>.edges.csv`` integer edge list into a symmetric adjacency."""
    fname = next(f for f in os.listdir(folder) if f.endswith(".edges.csv"))
    edges = []
    n = 0
    with open(os.path.join(folder, fname)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, j = (int(t) for t in line.split(","))
            edges.append((i, j))
            n = max(n, i + 1, j + 1)
    rows = np.array([e[0] for e in edges] + [e[1] for e in edges])
    cols = np.array([e[1] for e in edges] + [e[0] for e in edges])
    data = np.ones(len(rows))
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    adj.data[:] = 1.0  # collapse any duplicate edges to 1
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


# --------------------------------------------------------------------------- #
# Planetoid citation format (cora): ind.cora.*
# --------------------------------------------------------------------------- #
def _parse_index_file(path: str) -> list[int]:
    return [int(line.strip()) for line in open(path)]


def _load_citation(folder: str, name: str):
    """Load a Planetoid ``ind.<name>.*`` citation dataset (adj, features, labels, split)."""
    import networkx as nx

    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for nm in names:
        with open(os.path.join(folder, f"ind.{name}.{nm}"), "rb") as f:
            objects.append(pkl.load(f, encoding="latin1") if sys.version_info[0] >= 3
                           else pkl.load(f))
    x, y, tx, ty, allx, ally, graph = objects

    test_idx_reorder = _parse_index_file(os.path.join(folder, f"ind.{name}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.tocsr()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    # networkx 3.x: from_dict_of_lists + modern adjacency_matrix.
    g = nx.from_dict_of_lists(graph)
    adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    return adj, features, labels, idx_train, idx_val, idx_test


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_graph(name: str):
    """Return the dataset as a ``networkx.Graph`` (structure only).

    Sufficient for delta-hyperbolicity placement (plan section 3). Works for every
    vendored dataset.
    """
    import networkx as nx

    folder = _folder(name)
    if name == "cora":
        adj, *_ = _load_citation(folder, "cora")
    else:
        adj = _load_edge_list(folder)
    g = nx.from_scipy_sparse_array(adj)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def load(name: str) -> dict:
    """Return a full bundle: ``adj`` (csr), ``features``, ``labels`` (if any), splits.

    For edge-list datasets there are no node features/labels, so identity
    (structural) features are used and ``labels`` is ``None`` -- those graphs anchor
    the phase diagram via an unsupervised/reconstruction gap, not classification.
    """
    folder = _folder(name)
    if name == "cora":
        adj, features, labels, idx_train, idx_val, idx_test = _load_citation(folder, "cora")
        return {
            "adj": adj,
            "features": features,
            "labels": labels,
            "idx_train": np.asarray(idx_train),
            "idx_val": np.asarray(idx_val),
            "idx_test": np.asarray(idx_test),
            "n_classes": int(labels.max()) + 1,
            "task": "nc",
        }
    adj = _load_edge_list(folder)
    n = adj.shape[0]
    return {
        "adj": adj,
        "features": sp.eye(n, format="csr", dtype=np.float64),
        "labels": None,
        "task": "lp",  # link prediction / reconstruction
    }
