"""Optimization experiments for the paper (Section 'Experiments').

Deep-linear network on the kappa-Stereographic model (every layer the same
constant-curvature space), via the collapse identity

    h(x) = exp_o( W_{1:N} . log_o(x) ),   W_{1:N} = W_N ... W_1.

A single fixed tangent-space regression problem (xtil, ytil) is generated once per
seed and reused at every curvature via x = exp_o(xtil), y = exp_o(ytil), so
curvature enters ONLY through the geometry.  Two losses: the intrinsic
L_K = mean_i d_K(h,y)^2 and the Euclidean surrogate E = mean_i ||W_{1:N} xtil - ytil||^2.

Signed sectional curvature K (geoopt's k): K<0 hyperbolic, K=0 Euclidean, K>0
spherical.  At K=0 the model is exactly R^d and L_0 = E.

WHAT WE VALIDATE (honest, path A):
  E1  K->0 collapse + linear convergence          (Thm convergence, Prop surrogate)
  E2  near-optimum sharpness lambda*_K ~ S_K(R)^2, sign flip at K=0
                                                  (step-size mechanism, Cor positive)
  E3  delta-balancedness maintained along training (Lem trajectory hypothesis)
  E7  no spurious minima: balanced in-tube inits all reach the global value
                                                  (Thm landscape)
  GC  gradient correctness: autograd == finite differences

The clean witness of the step-size ceiling eta*_K = O(1/S_K(R)^2) is the LATE-PHASE
sharpness: as the residual D(t) -> 0 the smoothness constant collapses to
L -> S_K(R)^2 * mean||xi||^2 (Prop adaptive), so we measure the top Hessian
eigenvalue lambda*_K of the intrinsic loss AT the converged solution and compare to
S_K(R)^2.  We do NOT report an inflated whole-trajectory step size (that quantity is
dominated by the benign large-residual region and is nearly curvature-flat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import manifolds

# Signed curvatures: spherical (K>0) -> Euclidean (0) -> hyperbolic (K<0).
CURVATURES = (1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -4.0, -8.0)


def s_signed(K: float, R: float) -> float:
    """Rauch factor S_K(R) = sn_K(R)/R:  >1 for K<0, =1 at K=0, <1 for K>0."""
    if K == 0.0:
        return 1.0
    if K < 0.0:
        x = math.sqrt(-K) * R
        return math.sinh(x) / x
    x = math.sqrt(K) * R
    return math.sin(x) / x


def s_c(c: float, R: float) -> float:
    """Back-compat: Rauch factor by curvature magnitude c>=0 (hyperbolic K=-c)."""
    return s_signed(-abs(c), R)


@dataclass
class Config:
    # architecture / problem
    depth: int = 3
    width: int = 6
    n_samples: int = 300
    radius: float = 1.2
    teacher_scale: float = 0.5
    # seeds
    seed: int = 0
    n_seeds: int = 5
    # training to the optimum (for sharpness) — eta<2/lambda* keeps it stable
    train_eta: float = 0.02
    train_steps: int = 5000
    # fixed-step trajectory (for the collapse / convergence panel)
    traj_eta: float = 0.03
    traj_steps: int = 1500
    curvatures: tuple = field(default_factory=lambda: CURVATURES)
    # robustness scaling sweep (E2): S_K(R)^2 collapse across R and architecture
    sweep_radii: tuple = (0.8, 1.0, 1.2, 1.5)
    sweep_arch: tuple = ((3, 6),)              # (depth, width) pairs
    sweep_curvatures: tuple = (0.5, 0.0, -0.5, -1.0, -2.0, -4.0)
    # landscape (E7): random balanced in-tube inits
    landscape_inits: int = 12
    landscape_curvatures: tuple = (0.0, -1.0, -4.0)
    jobs: int = 1                              # parallel worker processes (server)
    dtype: torch.dtype = manifolds.DEFAULT_DTYPE


DescentConfig = Config  # back-compat alias


# --------------------------------------------------------------------------- #
# Problem + model
# --------------------------------------------------------------------------- #

def _make_problem(cfg, seed=None, width=None, radius=None):
    """Fixed tangent-space regression problem (xtil, ytil), seeded independent of K."""
    d = cfg.width if width is None else width
    R = cfg.radius if radius is None else radius
    m = cfg.n_samples
    g = torch.Generator().manual_seed(cfg.seed if seed is None else seed)
    z = torch.randn(m, d, generator=g, dtype=cfg.dtype)
    u = torch.rand(m, 1, generator=g, dtype=cfg.dtype)
    xtil = torch.nn.functional.normalize(z, dim=1) * (u * R)
    T = torch.randn(d, d, generator=g, dtype=cfg.dtype) * (cfg.teacher_scale / d ** 0.5)
    ytil = xtil @ T.T
    yn = ytil.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ytil = ytil * (yn.clamp(max=R) / yn)          # keep targets inside the ball
    return xtil, ytil


def _balanced_init(cfg, seed=None, depth=None, width=None, orthogonal=True, scale=1.0):
    """Scaled-orthogonal factors -> exactly delta-balanced at t=0 (when scale=1)."""
    N = cfg.depth if depth is None else depth
    d = cfg.width if width is None else width
    g = torch.Generator().manual_seed((cfg.seed if seed is None else seed) + 1)
    Ws = []
    for _ in range(N):
        a = torch.randn(d, d, generator=g, dtype=cfg.dtype)
        q, _ = torch.linalg.qr(a)
        Ws.append((q * scale).clone().requires_grad_(True))
    return Ws


def _end_to_end(Ws):
    A = Ws[0]
    for j in range(1, len(Ws)):
        A = Ws[j] @ A
    return A


def _intrinsic_from_A(A, xtil, ytil, manifold):
    h = manifold.expmap0(xtil @ A.T)
    y = manifold.expmap0(ytil)
    return (manifold.dist(h, y) ** 2).mean()


def _intrinsic(Ws, xtil, ytil, manifold):
    return _intrinsic_from_A(_end_to_end(Ws), xtil, ytil, manifold)


def _surrogate(Ws, xtil, ytil):
    A = _end_to_end(Ws)
    return ((xtil @ A.T - ytil) ** 2).sum(dim=1).mean()


def _balancedness_defect(Ws) -> float:
    defect = 0.0
    for j in range(len(Ws) - 1):
        m = Ws[j + 1].T @ Ws[j + 1] - Ws[j] @ Ws[j].T
        defect = max(defect, float(m.norm().item()))
    return defect


# --------------------------------------------------------------------------- #
# Core routines
# --------------------------------------------------------------------------- #

def _run_gd(Ws, xtil, ytil, manifold, eta, n_steps, track_balance=False, loss="intrinsic"):
    """Full-batch GD on the factors at fixed step; return (loss_hist, defect_hist)."""
    losses, defects = [], []
    for _ in range(n_steps):
        for w in Ws:
            if w.grad is not None:
                w.grad = None
        L = _intrinsic(Ws, xtil, ytil, manifold) if loss == "intrinsic" \
            else _surrogate(Ws, xtil, ytil)
        losses.append(float(L.item()))
        if track_balance:
            defects.append(_balancedness_defect(Ws))
        L.backward()
        with torch.no_grad():
            for w in Ws:
                w -= eta * w.grad
    return losses, defects


def _train_to_opt(cfg, manifold, seed=None, width=None, radius=None, depth=None):
    """Train to the optimum at a small step; return converged A*, xtil, ytil."""
    xtil, ytil = _make_problem(cfg, seed=seed, width=width, radius=radius)
    Ws = _balanced_init(cfg, seed=seed, depth=depth, width=width)
    for _ in range(cfg.train_steps):
        for w in Ws:
            if w.grad is not None:
                w.grad = None
        _intrinsic(Ws, xtil, ytil, manifold).backward()
        with torch.no_grad():
            for w in Ws:
                w -= cfg.train_eta * w.grad
    return _end_to_end(Ws).detach(), xtil, ytil


def _sharpness(A, xtil, ytil, manifold, iters=80):
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


def _convergence_exponent(losses) -> float:
    """Per-step linear-convergence exponent |b|: slope of log(loss) vs step."""
    import numpy as np
    y = np.log(np.asarray(losses) + 1e-16)
    x = np.arange(len(y))
    lo, hi = len(y) // 10, max(len(y) // 10 + 5, int(len(y) * 0.6))
    if hi - lo < 3:
        lo, hi = 0, len(y)
    return float(abs(np.polyfit(x[lo:hi], y[lo:hi], 1)[0]))


def _mean_ci(vals):
    """Mean and 95% CI half-width (normal approx) over seeds."""
    import numpy as np
    a = np.asarray(vals, dtype=float)
    m = float(a.mean())
    if a.size < 2:
        return m, 0.0
    return m, float(1.96 * a.std(ddof=1) / math.sqrt(a.size))


# --------------------------------------------------------------------------- #
# Parallel sharpness table (server): compute lambda*_K for many configs
# --------------------------------------------------------------------------- #

def _lambda_job(args):
    """Worker: train to the optimum and measure lambda*_K for each K in the job."""
    cfg, depth, width, R, seed, Ks = args
    torch.set_num_threads(1)
    out = {}
    for K in Ks:
        man = manifolds.stereographic(K)
        A, xt, yt = _train_to_opt(cfg, man, seed=seed, width=width, radius=R, depth=depth)
        out[K] = _sharpness(A, xt, yt, man)
    return (depth, width, R, seed), out


def _run_jobs(jobs, n_workers):
    """Map _lambda_job over jobs, in parallel if n_workers>1 (falls back to serial)."""
    if n_workers <= 1 or len(jobs) <= 1:
        return [_lambda_job(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(_lambda_job, jobs))


# --------------------------------------------------------------------------- #
# E1/E2/E3 — core figure
# --------------------------------------------------------------------------- #

def run_core(cfg: Config | None = None) -> dict:
    """Trajectories (collapse+convergence), near-optimum sharpness, balancedness."""
    cfg = cfg or Config()
    Ks = list(cfg.curvatures)
    seeds = [cfg.seed + s for s in range(cfg.n_seeds)]
    R = cfg.radius

    trajectories, defect_traj, exponents = {}, {}, {}
    sharp, sharp_ci = {}, {}
    rel_per_seed = {K: [] for K in Ks}          # lambda*_K / lambda*_0 within each seed

    # per-seed sharpness at every K (so ratios are computed within a seed); parallel.
    jobs = [(cfg, cfg.depth, cfg.width, R, sd, Ks) for sd in seeds]
    lam_by_seed = {sd: lam for (_, _, _, sd), lam in _run_jobs(jobs, cfg.jobs)}

    for K in Ks:
        sharp[K], sharp_ci[K] = _mean_ci([lam_by_seed[sd][K] for sd in seeds])
        for sd in seeds:
            l0 = lam_by_seed[sd].get(0.0)
            if l0:
                rel_per_seed[K].append(lam_by_seed[sd][K] / l0)

        man = manifolds.stereographic(K)
        # trajectory + balancedness at seed 0 (fixed step).
        xtil, ytil = _make_problem(cfg, seed=seeds[0])
        Ws = _balanced_init(cfg, seed=seeds[0])
        losses, defects = _run_gd(Ws, xtil, ytil, man, cfg.traj_eta, cfg.traj_steps,
                                  track_balance=True)
        trajectories[K] = [max(v, 1e-16) for v in losses]
        defect_traj[K] = defects
        exponents[K] = _convergence_exponent(trajectories[K])

    # surrogate control = K=0 intrinsic trajectory (exact collapse; same code path).
    man0 = manifolds.stereographic(0.0)
    xtil, ytil = _make_problem(cfg, seed=seeds[0])
    surr, _ = _run_gd(_balanced_init(cfg, seed=seeds[0]), xtil, ytil, man0,
                      cfg.traj_eta, cfg.traj_steps)
    exponents["surrogate"] = _convergence_exponent([max(v, 1e-16) for v in surr])

    s2 = {K: s_signed(K, R) ** 2 for K in Ks}
    rel_mean = {K: _mean_ci(rel_per_seed[K]) for K in Ks}

    return {
        "curvatures": Ks, "radius": R, "n_seeds": cfg.n_seeds,
        "trajectories": trajectories, "surrogate_traj": surr,
        "defect_traj": defect_traj, "exponents": exponents,
        "sharpness": sharp, "sharpness_ci": sharp_ci,
        "sharpness_rel": {K: rel_mean[K][0] for K in Ks},
        "sharpness_rel_ci": {K: rel_mean[K][1] for K in Ks},
        "s2": s2,
        "eta_pred": {K: (2.0 / sharp[K] if sharp[K] > 0 else float("nan")) for K in Ks},
    }


# --------------------------------------------------------------------------- #
# E2 robustness — the scaling-collapse across R and architecture (server sweep)
# --------------------------------------------------------------------------- #

def run_scaling(cfg: Config | None = None) -> dict:
    """For every (R, arch, seed, K): measure lambda*_K/lambda*_0 and pair with S_K(R)^2.

    The headline robustness result: all points collapse onto the identity line
    y = x across radii and architectures. Embarrassingly parallel -> server.
    """
    cfg = cfg or Config()
    seeds = [cfg.seed + s for s in range(cfg.n_seeds)]
    Ks = list(cfg.sweep_curvatures)
    jobs = [(cfg, depth, width, R, sd, Ks)
            for (depth, width) in cfg.sweep_arch
            for R in cfg.sweep_radii
            for sd in seeds]
    points = []   # dicts: R, depth, width, seed, K, s2, rel
    for (depth, width, R, sd), lam in _run_jobs(jobs, cfg.jobs):
        l0 = lam.get(0.0)
        if not l0:
            continue
        for K in Ks:
            points.append({
                "R": R, "depth": depth, "width": width, "seed": sd, "K": K,
                "s2": s_signed(K, R) ** 2, "rel": lam[K] / l0,
            })
    return {"points": points,
            "radii": list(cfg.sweep_radii), "arch": [list(a) for a in cfg.sweep_arch]}


# --------------------------------------------------------------------------- #
# E7 landscape — no spurious minima on the tube
# --------------------------------------------------------------------------- #

def run_landscape(cfg: Config | None = None) -> dict:
    """Many random BALANCED in-tube inits -> do all reach the global value?"""
    cfg = cfg or Config()
    out = {}
    for K in cfg.landscape_curvatures:
        man = manifolds.stereographic(K)
        xtil, ytil = _make_problem(cfg, seed=cfg.seed)
        finals = []
        for i in range(cfg.landscape_inits):
            g = torch.Generator().manual_seed(9000 + i)
            Ws = []
            for _ in range(cfg.depth):
                q, _ = torch.linalg.qr(
                    torch.randn(cfg.width, cfg.width, generator=g, dtype=cfg.dtype))
                sc = (0.6 + 0.5 * torch.rand(1, generator=g, dtype=cfg.dtype)).item()
                Ws.append((q * sc).clone().requires_grad_(True))
            _run_gd(Ws, xtil, ytil, man, cfg.train_eta, cfg.train_steps)
            finals.append(_intrinsic(Ws, xtil, ytil, man).item())
        import numpy as np
        finals = np.asarray(finals)
        out[K] = {"finals": finals.tolist(),
                  "frac_global": float((finals < 1e-3).mean()),
                  "max_final": float(finals.max())}
    return out


# --------------------------------------------------------------------------- #
# GC — gradient correctness (autograd vs central finite differences)
# --------------------------------------------------------------------------- #

def gradient_check(cfg: Config | None = None, eps=1e-6) -> dict:
    """Max relative error between autograd and central-difference gradients."""
    cfg = cfg or Config()
    out = {}
    for K in (0.0, -1.0, -4.0):
        man = manifolds.stereographic(K)
        xtil, ytil = _make_problem(cfg, seed=cfg.seed)
        Ws = _balanced_init(cfg, seed=cfg.seed)
        for w in Ws:
            if w.grad is not None:
                w.grad = None
        _intrinsic(Ws, xtil, ytil, man).backward()
        auto = [w.grad.clone() for w in Ws]
        max_rel = 0.0
        g = torch.Generator().manual_seed(123)
        for j, w in enumerate(Ws):
            for _ in range(20):                      # sample random entries
                a = int(torch.randint(0, w.shape[0], (1,), generator=g))
                b = int(torch.randint(0, w.shape[1], (1,), generator=g))
                with torch.no_grad():
                    w[a, b] += eps
                    lp = _intrinsic(Ws, xtil, ytil, man).item()
                    w[a, b] -= 2 * eps
                    lm = _intrinsic(Ws, xtil, ytil, man).item()
                    w[a, b] += eps
                fd = (lp - lm) / (2 * eps)
                denom = max(abs(fd), abs(float(auto[j][a, b])), 1e-8)
                max_rel = max(max_rel, abs(fd - float(auto[j][a, b])) / denom)
        out[K] = max_rel
    return out
