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

## Key experimental finding (first full run) -- the architecture telescopes

**The layerwise Riemannian network, as defined (bias-free, consecutive hidden
manifolds sharing $\kappa_{\mathrm{model}}$), collapses to a plain Euclidean MLP on
$\log_o^{\kappa_{\mathrm{data}}}(x)$. Curvature -- both data and model -- is cosmetic
for the learned function.** Verified to machine precision (below). Therefore the
measured generalization gap is curvature-blind; it responds only to depth.

Mechanism (two telescopings, both $\log\circ\exp=\mathrm{id}$ at the origin):
- **Data side:** the data is $x=\exp_o^{\kappa_{\mathrm{data}}}(z)$ and the model's first
  layer is $\log_o^{\kappa_{\mathrm{data}}}(x)=z$ exactly -> $\kappa_{\mathrm{data}}$ cancels.
- **Model side:** layer $j$ outputs $\exp_o^{\kappa_{\mathrm{model}}}(a_j)$ and layer $j{+}1$
  starts with $\log_o^{\kappa_{\mathrm{model}}}(\cdot)=a_j$ -> every hidden $\exp/\log$ pair
  cancels. (The caveat in `synthetic.py` that this "does not telescope for $N\ge2$" is
  **wrong**: consecutive layers share $\kappa_{\mathrm{model}}$, so they do.)

Evidence:
- At fixed depth, $\Lambda_N$ spans **10 orders of magnitude** as $\kappa_{\mathrm{model}}$
  varies ($1.5\times10^3 \to 6.4\times10^{13}$) while the gap is **flat** ($0.00446$).
- Model output is **identical to $\le2\times10^{-15}$** across $\kappa_{\mathrm{model}}\in[-16,-0.25]$
  at fixed weights/input.
- `gap` / `train_loss` / `test_loss` are **bit-identical** across the entire
  $\kappa_{\mathrm{data}}$ axis (raw + matched-loss sweeps).
- The P2 sweep's apparent gap--$\Lambda_N$ correlation ($\tau{=}0.52$) is **entirely the
  depth channel** (deeper Euclidean MLP overfits 64 samples); the curvature-driven
  $\Lambda_N$ variation moves the gap by $\sim0$.

Consequences:
- **Neither P1 nor P2 is empirically testable in this architecture.** $\Lambda_N$ is a
  product of worst-case Jacobian norms of maps that compose to the identity -- it
  varies hugely but has no bearing on the function or its gap.
- **What must change to make curvature non-cosmetic:** break the telescoping --
  Mobius **bias** ($\oplus b$ on-manifold, doesn't cancel), **per-layer-varying**
  curvature $\kappa^{(j)}$ ($\log_{\kappa_{j+1}}\!\circ\exp_{\kappa_j}\ne\mathrm{id}$), or a
  genuinely intrinsic op. These are exactly the bias-free scope limits the paper's
  own Experiments section flags, and the Theorem-10 generalization study it defers
  to future work.
- **Real data** has no data-side cancellation ($\log_o^{\kappa}(x_{\mathrm{real}})$ is a real
  warp), but hidden $\kappa_{\mathrm{model}}$ **still telescopes**, and with one
  $\delta$-estimate per dataset + a different loss unit, real points can only
  corroborate placement.
- **Do not tune the task until an effect "appears"** -- the effect is structurally
  absent, so any appearance would be an artifact of breaking the model, not a finding.

### Follow-up: Mobius-bias model (curvature made non-cosmetic) -- the bound is loose

Adding an on-manifold Mobius bias (`mobius_bias=True`, `curvature_bias.yaml`) breaks
the telescoping, so $\kappa_{\mathrm{model}}$ **does** enter the function (verified: output
changes with $\kappa_{\mathrm{model}}$; bias-free it did not). Honest result on the
measured gap (depth/width fixed, sweep $\kappa_{\mathrm{model}}$, $n_{\mathrm{train}}{=}64$):

- gap vs $\sqrt{|\kappa_{\mathrm{model}}|}$: **no significant dependence** (Spearman
  $\rho{=}0.02$, $p{=}0.85$); non-monotone, within noise.
- Decisive control ($\kappa{=}-0.25$ vs $-16$, same data/seed): **$\Lambda_N$ moves
  $4\times10^7$ while the gap moves $1.27\times$.** Curvature has a small real effect on
  the gap (~27% over the range) but $\Lambda_N$ overestimates its sensitivity by ~7
  orders of magnitude.

**Robustness check (pre-registered, `curvature_bias_strong.yaml`):** to rule out a
scale artifact, the regime was amplified ONCE -- radius 1.5 (deeper in the ball),
bias scale 0.1, 20 seeds for power. Result is the **same null, now well-powered**:
gap vs $\sqrt{|\kappa_{\mathrm{model}}|}$ has $\rho{=}-0.02$, $p{=}0.84$; across the full
$\kappa_{\mathrm{model}}$ range $\Lambda_N$ varies $1.6\times10^7$ while the gap varies
$1.10\times$ (non-monotone, within $\sim$1 SEM). The null is real, not underpowered.

**Conclusion: the curvature-adaptive bound is empirically very loose** -- $\Lambda_N$ is
not a quantitative predictor of the measured gap, consistent with plan section 6
(validate shape, not magnitude) and the overparameterized-regime looseness of
arXiv:2309.13658. **Empirical chase stopped here** (committed up front). The honest
paper story: theory + the optimization experiments + (a) the telescoping analysis and
(b) this well-powered loose-bound characterization -- NOT a measured gap that scales as
the bound predicts.

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

- [x] Install CUDA torch (`torch 2.6.0+cu124`, RTX 4070); headline sweep done (600 runs, ~51 min CPU -> `results/headline.parquet`). **Finding: the measured gap is flat at the noise floor (~5e-4) across all curvatures/depths while Lambda_N spans 2e3..2.7e10 -- the synthetic task is not yet gap-inducing (the open task-design decision below). P1 slope~0, collapse R^2<0, Kendall tau~0.**
- [x] Wire real-dataset loaders (`rnn/data/real.py`) -- clean networkx-3/scipy reimpl reusing the vendored QGCN *data* (the torch-1.1 loaders were replaced rather than patched, per plan section 0's "prefer geoopt/maintained stack")
- [x] Q-GCN / HGCN models (`rnn/models/qgcn.py`) for Corollary 11 / prediction P3 -- ported vendored QGCN to torch 2.6 (torchvision import patched out; `use_bias=False` to avoid the unstable pseudo-hyperboloid `mobius_add`); trains on Cora (test_acc~0.72), exposes additive P3 penalty `sum_l sqrt(|kappa^l|)`. Trained via `rnn/train_real.py`
- [x] delta-hyperbolicity placement of real datasets (`rnn/data/curvature.py::dataset_curvature`; verified on Cora kappa~-6.8, Disease tree kappa floored)
- [x] Real-data anchors on the phase diagram + held-out extrapolation (the decisiveness check, plan section 3) -- Cora anchors (depths 2-4) overlaid via `rnn.train_real.gather_anchors` (`results/real_anchors.parquet`); held-out fit on the grid minus the high-kappa x high-depth corner (rel-error reported in `RESULTS.md`). NB held-out error is meaningless while the synthetic gap is flat (see above).
- [x] Matched-train-loss gap variant (plan section 7) -- `sweeps/configs/headline_matched.yaml`; run (600 runs, ~3 min, early-stops fast). **Also flat in curvature** (P1 R^2=0, tau=-0.17): the gap is bimodal in depth parity, not driven by sqrt|kappa|. So matched-fit did NOT expose a curvature gap -- both variants confirm the task needs redesign, not just the stopping rule.
- [x] `RESULTS.md`: embedded figures + fitted constant, P1 slope/$R^2$, Kendall $\tau$ -- written by `analysis/run_analysis.py` to `results/RESULTS.md` (+ real-anchor table)
- [~] Optimization experiment (`rnn/optimization.py`, `analysis/run_optimization.py`): deep-linear hyperbolic descent on the Poincare ball. The **c->0 collapse** (panel a + `figure_collapse`) is reproduced correctly (surrogate defined as the c=0 intrinsic limit, sidestepping the geoopt factor-2 gyro-metric convention). **OPEN:** the maximum-stable-step `eta*` signature (panels b/c) comes out ~flat in this geoopt setup -- the measured gradient amplification (~1.17x over c in [0,8] at R=0.8) is far weaker than the paper's, so `eta*` does not reproduce the ~80x drop. Reproducing it needs the paper's exact Poincare-map convention / smoothness regime (curvature bookkeeping, plan section 9). Writes to `figures/` only; `paper/figure_descent.pdf` is untouched.

### Deferred decisions (plan section 11; resolve after the first full run)

- [ ] **Task design:** how strongly $\kappa_{\mathrm{data}}$ should drive the *measured* gap.
  A single matched-curvature layer is curvature-trivial because $\log \circ \exp = \mathrm{id}$;
  curvature enters via depth / $\kappa_{\mathrm{model}}$ (hence $\Lambda_N$) and the analytic
  Bishop--Gromov penalty. Sweeps must use $N \geq 2$. See `rnn/data/synthetic.py`.
- [ ] **y-axis:** depth $N$ vs hidden curvature $\kappa_{\mathrm{model}}$ (or ship both as appendix variants)
- [ ] **Headline figure:** phase diagram vs collapse plot (pick whichever reads cleaner with real data on it)
- [ ] **Primary gap definition:** raw gap vs matched-train-loss gap
- [ ] **Real datasets:** which make the cut (keep those with a reliable curvature estimate)
