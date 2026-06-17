# Curvature-adaptive generalization for layerwise Riemannian neural networks

Experiments validating the *shape* of the curvature-adaptive generalization
bound (Theorem 10 / Corollary 11 of `paper/main.tex`). Design doc: `notes/plan.md`.

The pipeline trains layerwise Riemannian networks on synthetic data of **known**
curvature, instruments the network spatial Lipschitz constant
$\Lambda_N = \prod_j \alpha_j P_j \beta_{j-1}$, evaluates the bound, and produces
the headline **curvature phase diagram** plus the appendix collapse / P1-slope /
Kendall-$\tau$ figures.

## Setup

The same clone runs on the laptop (CPU) and the server/PC (CUDA).

```bash
conda env create -f environment.yml      # creates env "riemannian-nn" (python 3.11 + pip deps)
conda activate riemannian-nn
pip install -e .                          # install the `rnn` package (editable)
git submodule update --init               # pull the vendored QGCN (datasets + delta-hyperbolicity)
```

### Compute targets

- **Laptop / CPU (default):** `requirements.txt` installs the CPU/MPS torch wheel.
  Nothing else to do.
- **NVIDIA server / gaming PC (the heavy sweep):** install the CUDA torch build
  *first*, then the rest:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
  pip install -r requirements.txt
  ```
  Geometry runs in **float64** (the Poincare model is unstable in float32), so
  Apple-Silicon MPS is not a target for the heavy run -- use CUDA or CPU.

## Run

```bash
# 1. Smoke test (CPU, < 5 s) -- proves train -> instrument -> analyze works:
python -m sweeps.run_sweep   --config sweeps/configs/smoke.yaml
python -m analysis.run_analysis --results results/smoke.parquet

# 2. Headline sweep (server) -- the dense grid, ~20-60 GPU-h, embarrassingly parallel:
python -m sweeps.run_sweep   --config sweeps/configs/headline.yaml
python -m analysis.run_analysis --results results/headline.parquet
```

Results land in `results/*.parquet` (one row per config x seed) and figures in
`figures/*.pdf`; both are gitignored and fully regenerable from the sweeps.

## Layout (mirrors `notes/plan.md` section 10)

| Path | Role |
|------|------|
| `rnn/data/`       | synthetic known-$\kappa$ generator, real loaders (deferred), delta-hyperbolicity |
| `rnn/models/`     | layerwise Riemannian NN / hyperbolic MLP; Q-GCN wrapper (deferred) |
| `rnn/instrument/` | $\Lambda_N$ estimation ($\alpha_j, \beta_j, P_j$ via power iteration) + bound eval |
| `rnn/analysis/`   | P1 slope, collapse fit, Kendall $\tau$, figures |
| `sweeps/`         | config-driven grid runner + YAML configs |
| `analysis/`       | `run_analysis.py` figure/summary driver |
| `figures/`, `results/` | generated artifacts (gitignored) |
| `external/QGCN/`  | vendored Q-GCN submodule (datasets, manifolds, delta-hyperbolicity) |

## Project status / TODO

Temporary working tracker (we'll delete this section once the work settles).
Task state otherwise lives only in commit history -- there is no separate tracker.

### Done (laptop, CPU smoke-tested)

- [x] Repo scaffold + conda/pip env (CPU laptop + CUDA-server switch)
- [x] Vendored QGCN submodule (`external/QGCN`: datasets + delta-hyperbolicity)
- [x] Synthetic known-curvature generator (`rnn/data/synthetic.py`)
- [x] Layerwise Riemannian NN / hyperbolic MLP (`rnn/models/layerwise.py`)
- [x] $\Lambda_N$ instrumentation -- $\alpha_j, \beta_j, P_j$ via power iteration -- + bound eval (`rnn/instrument/`)
- [x] Config-driven sweep + smoke/headline configs (`sweeps/`)
- [x] Analysis: P1 slope, collapse, Kendall $\tau$, phase diagram (`rnn/analysis/`)
- [x] End-to-end CPU smoke test passing (sweep -> parquet -> figures)

### To do (server / GPU)

- [ ] Install CUDA torch; run the headline sweep
- [ ] Patch QGCN for PyTorch 2.x; wire real-dataset loaders (`rnn/data/real.py`)
- [ ] Q-GCN / HGCN models (`rnn/models/qgcn.py`) for Corollary 11 / prediction P3
- [ ] delta-hyperbolicity placement of real datasets (Cora / Disease / Airport / Pubmed / WordNet)
- [ ] Real-data anchors on the phase diagram + held-out extrapolation (the decisiveness check, plan section 3)
- [ ] Matched-train-loss gap variant (plan section 7)
- [ ] `RESULTS.md`: embedded figures + fitted constant, P1 slope/$R^2$, Kendall $\tau$
- [ ] Regenerate the optimization figures (descent / step-size) for the appendix

### Deferred decisions (plan section 11; resolve after the first full run)

- [ ] **Task design:** how strongly $\kappa_{\mathrm{data}}$ should drive the *measured* gap.
  A single matched-curvature layer is curvature-trivial because $\log \circ \exp = \mathrm{id}$;
  curvature enters via depth / $\kappa_{\mathrm{model}}$ (hence $\Lambda_N$) and the analytic
  Bishop--Gromov penalty. Sweeps must use $N \geq 2$. See `rnn/data/synthetic.py`.
- [ ] **y-axis:** depth $N$ vs hidden curvature $\kappa_{\mathrm{model}}$ (or ship both as appendix variants)
- [ ] **Headline figure:** phase diagram vs collapse plot (pick whichever reads cleaner with real data on it)
- [ ] **Primary gap definition:** raw gap vs matched-train-loss gap
- [ ] **Real datasets:** which make the cut (keep those with a reliable curvature estimate)
