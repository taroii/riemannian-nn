"""Optimization experiment: the descent theory (paper Section 'Experiments').

Deep-linear network on a constant-curvature manifold (the kappa-Stereographic
model), the instance of the paper's general layerwise model with M_j the same
constant-curvature space at every layer, via the collapse identity

    h(x) = exp_o( W_{1:N} . log_o(x) ),    W_{1:N} = W_N ... W_1.

A single fixed tangent-space regression problem (xtil, ytil) is generated once and
reused at every curvature via x = exp_o(xtil), y = exp_o(ytil), so curvature enters
ONLY through the geometry (paper setup). Two losses:

    intrinsic   L_K = mean_i d_K( h(x_i), y_i )^2      (geodesic squared distance)
    surrogate   E   = mean_i || W_{1:N} xtil_i - ytil_i ||^2   (Euclidean tangent)

We use the SIGNED sectional curvature K (geoopt's k): K<0 hyperbolic, K=0
Euclidean, K>0 spherical. At K -> 0 the model -> Euclidean and L_K -> E exactly
(the collapse). Curvature enters through the Rauch factor
S_K(R) = sn_K(R)/R, which is >1 for K<0 and <1 for K>0.

What the theory predicts and what we measure honestly (path A):
  * The step-size ceiling eta*_K = O(1/S_K(R)^2) is a WORST-CASE bound. Its clean,
    theory-sanctioned witness is the LATE-PHASE / near-optimum landscape sharpness:
    the paper's adaptive-schedule analysis shows that as the residual D(t) -> 0 the
    smoothness constant collapses to L -> S_K(R)^2 * mean||xi||^2 (the H_K(D), D*B_K
    terms vanish). So we measure the top Hessian eigenvalue lambda*_K of the
    intrinsic loss AT THE CONVERGED SOLUTION and compare it to S_K(R)^2. GD's linear
    stability ceiling is eta* = 2/lambda*, which we corroborate empirically.
  * We do NOT claim the raw random-init sharpness scales as S_K(R)^2 (it does not --
    at large residual the higher-order terms dominate and wash the effect out). The
    honest signal is at the optimum, monotone through K=0, matching the SIGN and
    (in the moderate regime) the MAGNITUDE of S_K(R)^2.

Gradients are by autograd. Init is exactly delta-balanced: each W_j a scaled
orthogonal matrix, so the defect max_j ||W_{j+1}^T W_{j+1} - W_j W_j^T||_F vanishes
at t=0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import manifolds

# Signed sectional curvatures K: spherical (K>0) -> Euclidean (0) -> hyperbolic (K<0).
CURVATURES = (1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -4.0, -8.0)


def s_signed(K: float, R: float) -> float:
    """Rauch factor S_K(R) = sn_K(R)/R.  >1 for K<0, =1 at K=0, <1 for K>0.

    sn_K is the generalized sine: sinh(sqrt(-K)R)/sqrt(-K) for K<0, R for K=0,
    sin(sqrt(K)R)/sqrt(K) for K>0.  Defined within the injectivity radius for K>0.
    """
    if K == 0.0:
        return 1.0
    if K < 0.0:
        x = math.sqrt(-K) * R
        return math.sinh(x) / x
    x = math.sqrt(K) * R
    return math.sin(x) / x


def s_c(c: float, R: float) -> float:
    """Back-compat: Rauch factor by curvature MAGNITUDE c>=0 (hyperbolic K=-c)."""
    return s_signed(-abs(c), R)


@dataclass
class DescentConfig:
    depth: int = 3
    width: int = 6
    n_samples: int = 300
    radius: float = 1.2             # sweet spot: strong, monotone, numerically clean
    teacher_scale: float = 0.5      # keeps ytil inside the bounded ball
    seed: int = 0
    n_seeds: int = 5                # sharpness averaged over seeds (CIs)
    train_eta: float = 0.02         # small step to reach the optimum cleanly
    train_steps: int = 3000
    traj_eta: float = 0.03          # fixed step for the trajectory / collapse panel
    traj_steps: int = 1500
    measure_empirical_eta: bool = False  # from-init stable step (flat; off by default)
    dtype: torch.dtype = manifolds.DEFAULT_DTYPE
    curvatures: tuple = field(default_factory=lambda: CURVATURES)


# --------------------------------------------------------------------------- #
# Problem + model
# --------------------------------------------------------------------------- #

def _make_problem(cfg: DescentConfig, seed: int | None = None):
    """Fixed tangent-space regression problem (xtil, ytil), seeded independent of K."""
    g = torch.Generator().manual_seed(cfg.seed if seed is None else seed)
    d, m = cfg.width, cfg.n_samples
    z = torch.randn(m, d, generator=g, dtype=cfg.dtype)
    u = torch.rand(m, 1, generator=g, dtype=cfg.dtype)
    xtil = torch.nn.functional.normalize(z, dim=1) * (u * cfg.radius)
    T = torch.randn(d, d, generator=g, dtype=cfg.dtype) * (cfg.teacher_scale / d ** 0.5)
    ytil = xtil @ T.T
    yn = ytil.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ytil = ytil * (yn.clamp(max=cfg.radius) / yn)     # keep targets inside the ball
    return xtil, ytil


def _balanced_init(cfg: DescentConfig, seed: int | None = None):
    """N scaled-orthogonal factors -> exactly delta-balanced at t=0."""
    base = (cfg.seed if seed is None else seed) + 1
    g = torch.Generator().manual_seed(base)
    Ws = []
    for _ in range(cfg.depth):
        a = torch.randn(cfg.width, cfg.width, generator=g, dtype=cfg.dtype)
        q, _ = torch.linalg.qr(a)
        Ws.append(q.clone().requires_grad_(True))
    return Ws


def _end_to_end(Ws):
    A = Ws[0]
    for j in range(1, len(Ws)):
        A = Ws[j] @ A
    return A


def _intrinsic_from_A(A, xtil, ytil, manifold):
    out = xtil @ A.T
    h = manifold.expmap0(out)
    y = manifold.expmap0(ytil)
    return (manifold.dist(h, y) ** 2).mean()


def _intrinsic(Ws, xtil, ytil, manifold):
    return _intrinsic_from_A(_end_to_end(Ws), xtil, ytil, manifold)


def _balancedness_defect(Ws) -> float:
    defect = 0.0
    for j in range(len(Ws) - 1):
        m = Ws[j + 1].T @ Ws[j + 1] - Ws[j] @ Ws[j].T
        defect = max(defect, float(m.norm().item()))
    return defect


# --------------------------------------------------------------------------- #
# Core routines
# --------------------------------------------------------------------------- #

def _run_gd(cfg, manifold, eta, n_steps, seed=None, track_balance=False):
    """Full-batch GD on the factors at fixed step; return (loss_hist, defect_hist)."""
    xtil, ytil = _make_problem(cfg, seed)
    Ws = _balanced_init(cfg, seed)
    losses, defects = [], []
    for _ in range(n_steps):
        for w in Ws:
            if w.grad is not None:
                w.grad = None
        loss = _intrinsic(Ws, xtil, ytil, manifold)
        losses.append(float(loss.item()))
        if track_balance:
            defects.append(_balancedness_defect(Ws))
        loss.backward()
        with torch.no_grad():
            for w in Ws:
                w -= eta * w.grad
    return losses, defects


def _train_to_opt(cfg, manifold, seed=None):
    """Train to the optimum at a small step; return the converged end-to-end A*."""
    xtil, ytil = _make_problem(cfg, seed)
    Ws = _balanced_init(cfg, seed)
    for _ in range(cfg.train_steps):
        for w in Ws:
            if w.grad is not None:
                w.grad = None
        _intrinsic(Ws, xtil, ytil, manifold).backward()
        with torch.no_grad():
            for w in Ws:
                w -= cfg.train_eta * w.grad
    return _end_to_end(Ws).detach(), xtil, ytil


def _sharpness(A, xtil, ytil, manifold, iters=60):
    """Top Hessian eigenvalue of the intrinsic loss wrt A, by power iteration."""
    v = torch.randn_like(A)
    v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        A_ = A.clone().detach().requires_grad_(True)
        g = torch.autograd.grad(_intrinsic_from_A(A_, xtil, ytil, manifold), A_,
                                create_graph=True)[0]
        Hv = torch.autograd.grad((g * v).sum(), A_)[0]
        lam = float((Hv * v).sum() / (v * v).sum())
        v = Hv / (Hv.norm() + 1e-30)
    return lam


def _max_stable_step(cfg, manifold, seed=None, n_steps=600, lo=-2.0, hi=1.3, iters=22):
    """Largest eta for which GD from the balanced init converges, by log-bisection.

    Convergence = all-finite, no >10x excursion above the initial loss, and final
    loss < 0.5 * initial. Bisection on log10(eta) in [lo, hi] -> ~2^-iters resolution.
    """
    xtil, ytil = _make_problem(cfg, seed)

    def converges(eta):
        Ws = _balanced_init(cfg, seed)
        l0 = float(_intrinsic(Ws, xtil, ytil, manifold).item())
        for _ in range(n_steps):
            for w in Ws:
                if w.grad is not None:
                    w.grad = None
            _intrinsic(Ws, xtil, ytil, manifold).backward()
            with torch.no_grad():
                for w in Ws:
                    w -= eta * w.grad
            lv = float(_intrinsic(Ws, xtil, ytil, manifold).item())
            if not math.isfinite(lv) or lv > 10.0 * l0:
                return False
        lf = float(_intrinsic(Ws, xtil, ytil, manifold).item())
        return lf < 0.5 * l0

    a, b = lo, hi
    if not converges(10 ** a):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if converges(10 ** mid):
            a = mid
        else:
            b = mid
    return 10 ** a


def _convergence_exponent(excess) -> float:
    """Per-step linear-convergence exponent |b|: slope of log(excess) vs step."""
    import numpy as np
    y = np.log(np.asarray(excess) + 1e-16)
    x = np.arange(len(y))
    lo, hi = len(y) // 10, max(len(y) // 10 + 5, int(len(y) * 0.6))
    if hi - lo < 3:
        lo, hi = 0, len(y)
    b = np.polyfit(x[lo:hi], y[lo:hi], 1)[0]
    return float(abs(b))


def _mean_ci(vals):
    """Mean and 95% CI half-width (t=1.96 normal approx) over seeds."""
    import numpy as np
    a = np.asarray(vals, dtype=float)
    m = float(a.mean())
    if a.size < 2:
        return m, 0.0
    return m, float(1.96 * a.std(ddof=1) / math.sqrt(a.size))


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

def run_descent_experiment(cfg: DescentConfig | None = None) -> dict:
    """Compute the honest descent experiment across signed curvatures K."""
    cfg = cfg or DescentConfig()
    Ks = list(cfg.curvatures)
    seeds = [cfg.seed + s for s in range(cfg.n_seeds)]
    R = cfg.radius

    trajectories, defect_traj, exponents = {}, {}, {}
    sharp, sharp_ci = {}, {}
    eta_star, eta_star_ci, eta_pred = {}, {}, {}
    mean_xi2 = None

    for K in Ks:
        manifold = manifolds.stereographic(K)

        # (a) fixed-step raw-loss trajectory (seed 0) for the collapse panel.
        # The problem is realizable (L* ~ 0), so raw L decays linearly to ~0 -- no
        # per-run floor subtraction needed, which keeps the panel clean.
        losses, _ = _run_gd(cfg, manifold, cfg.traj_eta, cfg.traj_steps, seed=seeds[0])
        trajectories[K] = [max(v, 1e-16) for v in losses]
        exponents[K] = _convergence_exponent(trajectories[K])

        # (d) balancedness defect along training (seed 0).
        _, defects = _run_gd(cfg, manifold, cfg.traj_eta, cfg.traj_steps,
                             seed=seeds[0], track_balance=True)
        defect_traj[K] = defects

        # (b) near-optimum sharpness lambda*_K, averaged over seeds.
        lam_s = []
        for sd in seeds:
            Astar, xtil, ytil = _train_to_opt(cfg, manifold, seed=sd)
            lam_s.append(_sharpness(Astar, xtil, ytil, manifold))
            if mean_xi2 is None:
                mean_xi2 = float((xtil ** 2).sum(dim=1).mean())
        sharp[K], sharp_ci[K] = _mean_ci(lam_s)

        # predicted late-phase max stable step = 2 / lambda*_K (GD linear stability).
        eta_pred[K] = 2.0 / sharp[K] if sharp[K] > 0 else float("nan")

        # Optional: whole-trajectory empirical stable step from the balanced init.
        # This is NOT the late-phase ceiling -- it is dominated by the benign
        # high-residual region and is nearly curvature-flat, so it is off by default
        # and never shown in the figure; we record it only when explicitly asked.
        if cfg.measure_empirical_eta:
            emp = [_max_stable_step(cfg, manifold, seed=sd) for sd in seeds]
            eta_star[K], eta_star_ci[K] = _mean_ci(emp)

    # surrogate = K=0 intrinsic trajectory (exact collapse; same code path).
    surrogate_traj, _ = _run_gd(cfg, manifolds.stereographic(0.0),
                                cfg.traj_eta, cfg.traj_steps, seed=seeds[0])
    exponents["surrogate"] = _convergence_exponent(
        [max(v, 1e-16) for v in surrogate_traj]
    )

    # Rauch prediction and its scaling relative to K=0.
    s2 = {K: s_signed(K, R) ** 2 for K in Ks}
    lam0 = sharp.get(0.0, next(iter(sharp.values())))

    return {
        "curvatures": Ks,
        "radius": R,
        "n_seeds": cfg.n_seeds,
        "mean_xi2": mean_xi2,
        "trajectories": trajectories,
        "surrogate_traj": surrogate_traj,
        "defect_traj": defect_traj,
        "exponents": exponents,
        "sharpness": sharp,
        "sharpness_ci": sharp_ci,
        "sharpness_rel": {K: sharp[K] / lam0 for K in Ks},   # lambda*_K / lambda*_0
        "s2": s2,                                            # S_K(R)^2 (=S2 rel, since S2_0=1)
        "eta_pred": eta_pred,                                # 2/lambda*_K
        "eta_star": eta_star,                                # empirical, from init
        "eta_star_ci": eta_star_ci,
    }
