# Results: p2

Generated from `results/p2.parquet` (600 runs, 60 grid cells). Regenerate with `python -m analysis.run_analysis --results results/p2.parquet`.

## Headline numbers

- **Measured gap** Δ: mean=0.004312, std=0.00212, range=[0.001499, 0.01251]
- **P1** (gap ~ √|κ_data| at fixed arch): slope=nan, R²=nan, p=nan
- **Collapse** (gap vs bound predictor, all cells): global constant C=0.00183, R²=-0.053
- **Kendall τ** (full bound vs gap): τ=0.523, p=3.5e-09 over 60 cells
- **Held-out extrapolation** (fit off the high-κ×high-depth corner, predict it): 30 held-out cells, mean rel-error=0.264

## Figures

### collapse
![collapse](../figures/p2_collapse.pdf)

### lambda_collapse
![lambda_collapse](../figures/p2_lambda.pdf)

## Real-data anchors

Placed on the phase diagram at their measured δ-hyperbolicity curvature. NOTE: real gaps are cross-entropy (node classification); synthetic gaps are MSE (regression) — the anchors mark *curvature placement*, not directly comparable color. P3 penalty = Σ_l √|κ^l| (additive, Cor. 11).

| dataset | depth | √|κ_est| | gap (CE) | test_acc | P3 penalty |
|---|---|---|---|---|---|
| cora | 2 | 2.61 | 0.7976 | 0.750 | 2 |
| cora | 3 | 2.61 | 0.7937 | 0.728 | 3 |
| cora | 4 | 2.61 | 0.7136 | 0.755 | 4 |

