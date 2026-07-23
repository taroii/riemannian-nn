# Curvature and optimization of deep linear networks on Riemannian manifolds

Experiments for the paper *The Influence of Curvature on Deep Linear Networks on
Riemannian Manifolds* (`paper/main.tex`). We study how the sectional curvature of
the target manifold affects the optimization of a deep linear network trained
under the **intrinsic** squared-geodesic loss versus the Euclidean **tangent-space
surrogate**.

The network is the collapse-form deep-linear map on the κ-Stereographic model,

```
h(x) = exp_o( W_{1:N} · log_o(x) ),   W_{1:N} = W_N ... W_1,
```

with a single fixed tangent-space regression problem lifted through the geometry
at each curvature, so **curvature enters only through the geometry, never through
the data**.

## What the experiments show

Signed sectional curvature `K` (K<0 hyperbolic, K=0 Euclidean, K>0 spherical):

1. **K→0 collapse + linear convergence.** At K=0 the intrinsic loss equals the
   Euclidean surrogate exactly; gradient descent converges linearly at every K.
2. **The step-size mechanism.** The theorem's step-size ceiling
   `η*_K = O(1/S_K(R)²)` is a worst-case bound whose clean witness is the
   **near-optimum landscape sharpness** `λ*_K` (top Hessian eigenvalue of the
   intrinsic loss at the converged solution) — where the paper's adaptive-schedule
   analysis gives `L → S_K(R)²·mean‖ξ‖²`. It **increases with hyperbolic curvature
   and decreases for spherical curvature** (Cor.: positive curvature *relaxes* the
   step), matching `S_K(R)²` in the moderate regime and exceeding it at strong
   hyperbolic curvature (the `H_K, B_K` terms).
3. **δ-balancedness** stays small along training at every curvature, so the
   structural hypothesis of the descent lemma is maintained.

> **Honest scope.** `S_K(R)²` is a *worst-case* factor. The whole-trajectory stable
> step from a high-residual initialization is nearly curvature-flat (the benign
> region dominates); the curvature penalty is a *late-phase* effect, which is why we
> measure sharpness at the optimum. We validate the shape and sign of the theory,
> not an inflated magnitude.

## Setup

```bash
conda env create -f environment.yml      # creates env "riemannian-nn" (python 3.11 + pip deps)
conda activate riemannian-nn
```

Geometry runs in **float64** (the stereographic model is unstable in float32). The
experiment is small and CPU-only (no GPU needed).

## Run

```bash
python run.py
```

Writes `figure_descent.pdf`, `figure_collapse.pdf` and `RESULTS.md`. All generated artifacts
are gitignored and fully regenerable.

