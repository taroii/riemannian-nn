# Hyperbolic vs Euclidean graph embedding

Experiments for the layerwise Riemannian neural network paper (`paper/main.tex`).
This is the *positive* experiment: a hierarchy (tree) embeds into a **Poincaré ball**
with far lower distortion, near-perfect reconstruction, and better generalization than
a **Euclidean** embedding at the same dimension — the representational-capacity
advantage that motivates the architecture.

## Setup

```bash
conda env create -f environment.yml      # creates env "riemannian-nn" (python 3.11 + pip deps)
conda activate riemannian-nn
pip install -e .                          # install the `rnn` package (editable)
```

Geometry runs in **float64** (the Poincaré model is unstable in float32). The
experiment is small and CPU-only (no GPU needed). On a CUDA box, install the matching
torch build first (`pip install torch --index-url https://download.pytorch.org/whl/cuXXX`)
before `pip install -r requirements.txt`.

## Run

```bash
python -m analysis.run_tree_embedding
```

Writes `results/tree_embedding.parquet`, `results/tree_generalization.parquet`, the
figures below, and `results/RESULTS_tree_embedding.md`. Results/figures are gitignored
and fully regenerable.

## Layout

| Path | Role |
|------|------|
| `rnn/manifolds.py` | curvature conventions + geoopt Stereographic construction |
| `rnn/data/tree_task.py` | tree / hierarchy structure generators |
| `rnn/embed.py` | Euclidean vs Poincaré-ball embedding + distortion / mAP metrics |
| `rnn/models/euclidean.py` | the naive Euclidean MLP baseline |
| `rnn/analysis/figures.py` | figures (distortion/mAP vs dim, generalization, Shepard) |
| `analysis/run_tree_embedding.py` | experiment driver |
| `figures/`, `results/` | generated artifacts (gitignored) |

## Results (balanced binary tree, 127 nodes)

**Representation — distortion / reconstruction mAP vs embedding dim** (`tree_embedding.pdf`):

| dim | Euclid distortion / mAP | Hyperbolic distortion / mAP |
|-----|-------------------------|------------------------------|
| 2   | 0.228 / 0.27            | 0.36 / 0.26  (both hard*)     |
| 3   | 0.160 / 0.43            | **0.012 / 0.99**             |
| 5   | 0.106 / 0.61            | **0.010 / 1.00**             |
| 10  | 0.070 / 0.73            | **0.009 / 1.00**             |

From `dim >= 3` hyperbolic embeds the tree with ~10x lower distortion and near-perfect
reconstruction, while Euclidean never gets there even at dim 10. *`dim == 2` is hard
for gradient methods in both spaces (the 2D Poincaré disk needs embeddings at
machine-precision near the boundary; Sarkar's low-distortion 2D construction is
combinatorial, not gradient-based).

**Generalization — held-out distortion vs fraction of pairs held out, dim 10**
(`tree_generalization.pdf`): fit the embedding to a subset of node-pair distances,
evaluate on held-out pairs.

| fraction held out | Euclid held-out | Hyperbolic held-out |
|-------------------|-----------------|----------------------|
| 0.3 | 0.091 | **0.016** |
| 0.5 | 0.095 | **0.031** |
| 0.7 | 0.111 | **0.047** |
| 0.9 | 0.283 | **0.200** |

Hyperbolic infers unseen distances from partial structure 3-6x better — it generalizes
the hierarchy, not just fits it.

**Shepard diagram, dim 10** (`tree_shepard.pdf`): scaled embedding distance vs true
graph distance. Hyperbolic points hug the diagonal (faithful metric, distortion ~0.01);
Euclidean scatter widely (~0.07).
