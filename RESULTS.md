# Results: curvature and the deep-linear step size

kappa-Stereographic deep-linear net, depth 3, width 6, m=300, 50 seeds. Signed curvature K (<0 hyperbolic, >0 spherical). Regenerate: `python run.py` (quick) / `python run.py --full` (server).

## E1 collapse + linear convergence  (Thm convergence, Prop surrogate)
- intrinsic |b| at K=0 = 0.0041262; surrogate |b| = 0.0041262  (identical => exact collapse)

![descent](figure_descent.pdf)

## E2 near-optimum sharpness ~ S_K(R)^2  (step-size mechanism, Cor positive)

| K | S_K(R)^2 | lambda*_K | lambda*_K/lambda*_0 |
|---|---|---|---|
| -8 | 19.213 | 23.2245 (±6.0225) | 28.843 (±7.645) |
| -6 | 10.283 | 9.9733 (±2.1028) | 12.374 (±2.644) |
| -4 | 5.187 | 4.0335 (±0.5551) | 5.002 (±0.693) |
| -2 | 2.415 | 1.6773 (±0.1110) | 2.078 (±0.135) |
| -1 | 1.582 | 1.1254 (±0.0386) | 1.394 (±0.042) |
| -0.5 | 1.264 | 0.9428 (±0.0214) | 1.167 (±0.016) |
| 0 | 1.000 | 0.8080 (±0.0151) | 1.000 (±0.000) |
| 0.5 | 0.782 | 0.7652 (±0.0147) | 0.947 (±0.003) |
| 1 | 0.603 | 0.7377 (±0.0140) | 0.913 (±0.005) |

Robustness (fig:scaling): across radii [0.8, 1.0, 1.2, 1.5] and architectures [[2, 4], [3, 6], [4, 10]], the ratio measured/predicted has geometric mean nan and 93% of 4200 points fall within 2x of the S_K(R)^2 prediction.

![scaling](figure_scaling.pdf)

## E7 no spurious minima on the tube  (Thm landscape)

| K | frac reaching global (loss<1e-3) | max final loss |
|---|---|---|
| -4 | 100% | 3.20e-04 |
| -1 | 100% | 1.72e-04 |
| 0 | 98% | 1.89e-03 |

![landscape](figure_landscape.pdf)

## GC gradient correctness (autograd vs central differences)
- max relative error: K=0: 1.2e-08, K=-1: 1.6e-08, K=-4: 2.8e-07

