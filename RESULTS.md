# Results: curvature and the deep-linear step size

kappa-Stereographic deep-linear net, depth 3, width 6, m=300, 50 seeds. Signed curvature K (<0 hyperbolic, >0 spherical). Regenerate: `python run.py` (quick) / `python run.py --full` (server).

## E1 collapse + linear convergence  (Thm convergence, Prop surrogate)
- intrinsic |b| at K=0 = 0.0041262; surrogate |b| = 0.0041262  (identical => exact collapse)

![descent](figure_descent.pdf)

## E2 near-optimum sharpness ~ S_K(R)^2  (step-size mechanism, Cor positive)

| K | S_K(R)^2 | lambda*_K | lambda*_K/lambda*_0 |
|---|---|---|---|
| -8 | 19.213 | 23.2145 (±6.0117) | 28.832 (±7.631) |
| -6 | 10.283 | 9.9768 (±2.1065) | 12.379 (±2.649) |
| -4 | 5.187 | 4.0355 (±0.5552) | 5.005 (±0.693) |
| -2 | 2.415 | 1.6778 (±0.1111) | 2.079 (±0.135) |
| -1 | 1.582 | 1.1251 (±0.0387) | 1.393 (±0.042) |
| -0.5 | 1.264 | 0.9431 (±0.0214) | 1.167 (±0.016) |
| 0 | 1.000 | 0.8080 (±0.0151) | 1.000 (±0.000) |
| 0.5 | 0.782 | 0.7641 (±0.0146) | 0.946 (±0.004) |
| 1 | 0.603 | 0.7375 (±0.0141) | 0.913 (±0.005) |

Robustness (fig:scaling): across radii [0.8, 1.0, 1.2, 1.5] and architectures [[2, 4], [3, 6], [4, 10]], in the controlled regime sqrt(|K|)*R <= 2 the measured/predicted ratio has geometric mean 0.953 with 94% of 3300 configs within 1.5x of the S_K(R)^2 prediction; over the full range 93% of 4198 fall within 2x, degrading gracefully toward the injectivity boundary as the higher-order H_K,B_K terms enter (0 boundary configs produced non-finite float64 geometry and were excluded).

![scaling](figure_scaling.pdf)

## E7 no spurious minima on the tube  (Thm landscape)

| K | frac reaching global (loss<1e-3) | max final loss |
|---|---|---|
| -4 | 100% | 2.86e-04 |
| -1 | 100% | 1.47e-04 |
| 0 | 100% | 1.23e-04 |

![landscape](figure_landscape.pdf)

## GC gradient correctness (autograd vs central differences)
- max relative error: K=0: 1.1e-08, K=-1: 1.4e-08, K=-4: 1.7e-07

