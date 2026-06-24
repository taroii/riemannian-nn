# Results: tree embedding (hyperbolic vs Euclidean)

Graph: balanced_tree_r2_h6 (127 nodes, diameter 12). 3 seeds. Regenerate: `python -m analysis.run_tree_embedding`.

## Fitting: distortion / reconstruction vs embedding dim

Hyperbolic embeddings represent the hierarchy with far lower distortion and near-perfect reconstruction at small dimension -- the representational-capacity advantage the Riemannian architecture is built for. (dim=2 is hard for gradient methods in both spaces.)

| space | dim | distortion | recon mAP |
|---|---|---|---|
| euclid | 2 | 0.2281 | 0.272 |
| euclid | 3 | 0.1596 | 0.427 |
| euclid | 5 | 0.1056 | 0.612 |
| euclid | 10 | 0.0696 | 0.727 |
| hyper | 2 | 0.3027 | 0.309 |
| hyper | 3 | 0.0136 | 0.983 |
| hyper | 5 | 0.0098 | 1.000 |
| hyper | 10 | 0.0093 | 1.000 |

![tree_embedding](../figures/tree_embedding.pdf)

## Generalization: held-out distortion vs fraction held out (dim=10)

Embeddings fit to a subset of node-pair distances, evaluated on held-out pairs. Hyperbolic infers unseen distances from partial structure far better than Euclidean -- the generalization claim with teeth.

| space | holdout frac | held-out distortion |
|---|---|---|
| euclid | 0.3 | 0.0900 |
| euclid | 0.5 | 0.0941 |
| euclid | 0.7 | 0.1140 |
| euclid | 0.9 | 0.2787 |
| hyper | 0.3 | 0.0164 |
| hyper | 0.5 | 0.0288 |
| hyper | 0.7 | 0.0517 |
| hyper | 0.9 | 0.1907 |

![tree_generalization](../figures/tree_generalization.pdf)

