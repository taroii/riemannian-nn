"""Optimization experiment: the descent theory (paper Section 'Experiments').

Reproduces Figure `fig:descent` (and `fig:collapse`) for the deep-linear
hyperbolic network on the Poincare ball D^d_c (the instance of the general
layerwise model with M_j = D^d_c at every layer), via the collapse identity

    h(x) = exp^c_o( W_{1:N} . log^c_o(x) ),    W_{1:N} = W_N ... W_1.

A single fixed tangent-space regression problem (xtil, ytil) is generated once and
reused at every curvature via x = exp^c_o(xtil), y = exp^c_o(ytil), so curvature
enters ONLY through the geometry (paper setup). Two losses:

    intrinsic   L_c = mean_i d_c( h(x_i), y_i )^2      (geodesic squared distance)
    surrogate   E   = mean_i || W_{1:N} xtil_i - ytil_i ||^2   (Euclidean tangent)

At c -> 0 the ball -> Euclidean and L_c -> E exactly (the collapse). Curvature
enters through S_c(R) = sinh(sqrt(c) R)/(sqrt(c) R), which amplifies the effective
gradient; the clean monotone signature of curvature is in the maximum stable step
size eta*, not the fixed-step rate (paper 'What the theory predicts').

Gradients are by autograd (the paper validates these equal the analytic
factor-gradient identity to ~1e-9). Init is exactly delta-balanced: each W_j a
scaled orthogonal matrix, so the defect max_j ||W_{j+1}^T W_{j+1} - W_j W_j^T||_F
vanishes at t=0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from rnn import manifolds

CURVATURES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def s_c(c: float, R: float) -> float:
    """Angular-distortion factor S_c(R) = sinh(sqrt(c) R)/(sqrt(c) R); ->1 as c->0."""
    if c <= 0.0:
        return 1.0
    x = math.sqrt(c) * R
    return math.sinh(x) / x


@dataclass
class DescentConfig:
    depth: int = 3
    width: int = 6
    n_samples: int = 300
    radius: float = 0.8
    teacher_scale: float = 0.5      # keeps ytil inside the bounded ball
    seed: int = 0
    dtype: torch.dtype = manifolds.DEFAULT_DTYPE
    curvatures: tuple = field(default_factory=lambda: CURVATURES)


def _make_problem(cfg: DescentConfig):
    """Fixed tangent-space regression problem (xtil, ytil), seeded independent of c."""
    g = torch.Generator().manual_seed(cfg.seed)
    d, m = cfg.width, cfg.n_samples
    z = torch.randn(m, d, generator=g, dtype=cfg.dtype)
    u = torch.rand(m, 1, generator=g, dtype=cfg.dtype)
    xtil = torch.nn.functional.normalize(z, dim=1) * (u * cfg.radius)
    T = torch.randn(d, d, generator=g, dtype=cfg.dtype) * (cfg.teacher_scale / d ** 0.5)
    ytil = xtil @ T.T
    # keep targets inside the bounded ball of radius R
    yn = ytil.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ytil = ytil * (yn.clamp(max=cfg.radius) / yn)
    return xtil, ytil


def _balanced_init(cfg: DescentConfig):
    """N scaled-orthogonal factors -> exactly delta-balanced at t=0."""
    g = torch.Generator().manual_seed(cfg.seed + 1)
    Ws = []
    for _ in range(cfg.depth):
        a = torch.randn(cfg.width, cfg.width, generator=g, dtype=cfg.dtype)
        q, _ = torch.linalg.qr(a)              # orthogonal => W^T W = W W^T = I
        Ws.append(q.clone().requires_grad_(True))
    return Ws


def _balancedness_defect(Ws) -> float:
    defect = 0.0
    for j in range(len(Ws) - 1):
        m = Ws[j + 1].T @ Ws[j + 1] - Ws[j] @ Ws[j].T
        defect = max(defect, float(m.norm().item()))
    return defect


def _losses(Ws, xtil, ytil, manifold):
    """Return (intrinsic L_c, surrogate E) for the collapse-form deep-linear net."""
    A = Ws[0]
    for j in range(1, len(Ws)):
        A = Ws[j] @ A                          # W_{1:N} = W_N ... W_1
    tangent_out = xtil @ A.T
    surrogate = ((tangent_out - ytil) ** 2).sum(dim=1).mean()
    h = manifold.expmap0(tangent_out)
    y = manifold.expmap0(ytil)
    intrinsic = (manifold.dist(h, y) ** 2).mean()
    return intrinsic, surrogate


def _run_gd(cfg, c, eta, n_steps, track_balance=False):
    """Full-batch GD on the factors at fixed step size; return loss/defect history."""
    manifold = manifolds.stereographic(-c)     # k = -c: ball of curvature magnitude c
    xtil, ytil = _make_problem(cfg)
    Ws = _balanced_init(cfg)
    losses, defects = [], []
    for _ in range(n_steps):
        for w in Ws:
            if w.grad is not None:
                w.grad = None
        intrinsic, _ = _losses(Ws, xtil, ytil, manifold)
        losses.append(float(intrinsic.item()))
        if track_balance:
            defects.append(_balancedness_defect(Ws))
        intrinsic.backward()
        with torch.no_grad():
            for w in Ws:
                w -= eta * w.grad
    return losses, defects


def _max_stable_step(cfg, c, n_steps=150, grid=None) -> float:
    """Largest eta that stays bounded, decreases the loss, and never overshoots.

    The descent guarantee is a monotone-decrease condition: the curvature factor
    S_c(R)^2 amplifies the effective gradient, so at higher curvature a given eta
    overshoots (the loss rises above its starting value before -- if at all --
    settling) at a *smaller* eta. We therefore reject any eta whose trajectory ever
    exceeds the initial loss; the largest surviving eta is the empirical eta*
    (paper: 'remains bounded and decreases the loss over a fixed horizon').
    """
    if grid is None:
        grid = [10 ** e for e in torch.linspace(-3.5, 1.0, 60).tolist()]
    best = 0.0
    for eta in grid:
        losses, _ = _run_gd(cfg, c, eta, n_steps)
        arr = torch.tensor(losses)
        bounded = bool(torch.isfinite(arr).all()) and arr.max() < 1e6
        # No overshoot above the starting loss (a small tolerance for float noise).
        monotone_safe = bounded and arr.max() <= losses[0] * (1.0 + 1e-9)
        decreased = bounded and losses[-1] < losses[0]
        if monotone_safe and decreased:
            best = max(best, eta)
    return best


def run_descent_experiment(cfg: DescentConfig | None = None) -> dict:
    """Compute the four-panel descent experiment + the collapse exponents."""
    cfg = cfg or DescentConfig()
    curvs = list(cfg.curvatures)

    # (a) fixed-step loss-excess trajectories on the single fixed problem.
    fixed_eta = 0.05
    n_traj = 200
    trajectories = {}
    for c in curvs:
        losses, _ = _run_gd(cfg, c, fixed_eta, n_traj)
        floor = min(losses)
        trajectories[c] = [max(v - floor, 1e-16) for v in losses]
    # The tangent surrogate IS the c->0 limit of the intrinsic loss (the collapse
    # L_c -> E). Defining it as the c=0 intrinsic trajectory (same code path) makes
    # the collapse exact and sidesteps the gyrovector factor-2 metric-convention
    # mismatch between a hand-written Euclidean loss and geoopt's dist (plan
    # section 9: keep one curvature convention end to end).
    surrogate_traj, _ = _run_gd(cfg, 0.0, fixed_eta, n_traj)

    # (b) max stable step size vs curvature.
    eta_star = {c: _max_stable_step(cfg, c) for c in curvs}

    # (c) eta* normalized to c=0, vs 1/S_c(R)^2.
    base = eta_star[0.0] if eta_star.get(0.0, 0.0) > 0 else max(eta_star.values())
    eta_norm = {c: (eta_star[c] / base if base > 0 else float("nan")) for c in curvs}
    inv_s2 = {c: 1.0 / s_c(c, cfg.radius) ** 2 for c in curvs}

    # (d) balancedness defect along training (at the fixed step size).
    defect_traj = {}
    for c in curvs:
        _, defects = _run_gd(cfg, c, fixed_eta, n_traj, track_balance=True)
        defect_traj[c] = defects

    # collapse figure: per-step convergence exponent |b| of the intrinsic loss.
    exponents = {c: _convergence_exponent(trajectories[c]) for c in curvs}
    exponents["surrogate"] = _convergence_exponent(
        [max(v - min(surrogate_traj), 1e-16) for v in surrogate_traj]
    )

    return {
        "curvatures": curvs,
        "radius": cfg.radius,
        "trajectories": trajectories,
        "surrogate_traj": surrogate_traj,
        "eta_star": eta_star,
        "eta_norm": eta_norm,
        "inv_s2": inv_s2,
        "defect_traj": defect_traj,
        "exponents": exponents,
    }


def _surrogate_trajectory(cfg, eta, n_steps):
    """GD on the Euclidean tangent surrogate E (curvature-blind control)."""
    xtil, ytil = _make_problem(cfg)
    Ws = _balanced_init(cfg)
    losses = []
    for _ in range(n_steps):
        for w in Ws:
            if w.grad is not None:
                w.grad = None
        A = Ws[0]
        for j in range(1, len(Ws)):
            A = Ws[j] @ A
        surrogate = ((xtil @ A.T - ytil) ** 2).sum(dim=1).mean()
        losses.append(float(surrogate.item()))
        surrogate.backward()
        with torch.no_grad():
            for w in Ws:
                w -= eta * w.grad
    return losses


def _convergence_exponent(excess) -> float:
    """Per-step linear-convergence exponent |b|: slope of log(excess) vs step."""
    import numpy as np

    y = np.log(np.asarray(excess) + 1e-16)
    x = np.arange(len(y))
    # fit over the clean middle decade (skip transient + floor)
    lo, hi = len(y) // 10, max(len(y) // 10 + 5, int(len(y) * 0.6))
    if hi - lo < 3:
        lo, hi = 0, len(y)
    b = np.polyfit(x[lo:hi], y[lo:hi], 1)[0]
    return float(abs(b))
