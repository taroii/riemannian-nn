# Results: headline_matched

Generated from `results/headline_matched.parquet` (600 runs, 60 grid cells). Regenerate with `python -m analysis.run_analysis --results results/headline_matched.parquet`.

## Headline numbers

- **Measured gap** Δ: mean=7.331e-05, std=0.001386, range=[-0.005428, 0.003363]
- **P1** (gap ~ √|κ_data| at fixed arch): slope=-8.45e-12, R²=0.000, p=1
- **Collapse** (gap vs bound predictor, all cells): global constant C=1.921e-05, R²=-0.020
- **Kendall τ** (full bound vs gap): τ=-0.169, p=0.057 over 60 cells
- **Held-out extrapolation** (fit off the high-κ×high-depth corner, predict it): 8 held-out cells, mean rel-error=1.08

## Figures

### phase_diagram
![phase_diagram](../figures/headline_matched_phase.pdf)

### phase_diagram_3d
![phase_diagram_3d](../figures/headline_matched_phase_3d.png)

### collapse
![collapse](../figures/headline_matched_collapse.pdf)

### p1_slope
![p1_slope](../figures/headline_matched_p1.pdf)

### lambda_collapse
![lambda_collapse](../figures/headline_matched_lambda.pdf)

## Real-data anchors

Placed on the phase diagram at their measured δ-hyperbolicity curvature. NOTE: real gaps are cross-entropy (node classification); synthetic gaps are MSE (regression) — the anchors mark *curvature placement*, not directly comparable color. P3 penalty = Σ_l √|κ^l| (additive, Cor. 11).

| dataset | depth | √|κ_est| | gap (CE) | test_acc | P3 penalty |
|---|---|---|---|---|---|
| cora | 2 | 2.61 | 0.7976 | 0.750 | 2 |
| cora | 3 | 2.61 | 0.7937 | 0.728 | 3 |
| cora | 4 | 2.61 | 0.7136 | 0.755 | 4 |

