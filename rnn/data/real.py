"""Real-dataset loaders (plan section 5) -- DEFERRED to the server run.

Disease, Airport, Cora/Citeseer/Pubmed and a WordNet-derived hierarchy are
bundled in the vendored QGCN/HGCN stack (``external/QGCN/data``). QGCN's loaders
target torch==1.1 (see ``external/QGCN/requirements.txt``) and will need PyTorch
2.x patching -- that work belongs on the compute box, not the laptop smoke test.

This module intentionally only locates the data and fails loudly with guidance,
so importing :mod:`rnn.data` never drags in the rotten QGCN import chain.
"""

from __future__ import annotations

import os

# Datasets vendored inside the QGCN submodule.
QGCN_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "external",
    "QGCN",
    "data",
)

# Real graph datasets to anchor on the phase diagram (plan section 3), with the
# QGCN folder name where present. Curvature placement uses delta-hyperbolicity
# (see rnn.data.curvature).
REAL_DATASETS = {
    "cora": "cora",
    "disease": "disease_md",
    # "airport": ...,   # add when wiring the Airport loader on the server
    # "citeseer": ...,
    # "pubmed": ...,
    # "wordnet": ...,   # Nickel & Kiela Poincare-embeddings hierarchy
}


def available() -> dict[str, str]:
    """Return ``{name: abs_path}`` for datasets physically present in the submodule."""
    out = {}
    for name, folder in REAL_DATASETS.items():
        path = os.path.join(QGCN_DATA_ROOT, folder)
        if os.path.isdir(path):
            out[name] = path
    return out


def load(name: str):
    raise NotImplementedError(
        "Real-dataset loaders are deferred to the server run (QGCN targets "
        "torch==1.1 and needs PyTorch-2.x patching). Datasets are vendored at "
        f"{QGCN_DATA_ROOT}. Available now: {list(available())}. Wire this on the "
        "GPU box, reusing external/QGCN/utils/data_utils.py."
    )
