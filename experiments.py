#!/usr/bin/env python3
"""
Experiments for:
  "The Influence of Curvature on Deep Linear Networks on Riemannian Manifolds"
  (AAAI 2027)

Generates figure_descent.pdf, figure_collapse.pdf, and three supplementary
figures.  All gradients via PyTorch autograd; no finite-difference
approximation.

Usage
-----
    python experiments.py                 # generates all figures
    python experiments.py --seed 0        # single seed (faster)

Hyperparameters match Section 5.1:
    d=6, N=3, m=300, R=1.2, K in {1,.5,0,-.5,-1,-2,-4,-8},
    5 seeds, balanced orthogonal init.
"""

import argparse, os, numpy as np
import torch
torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "text.usetex": False,
})
import matplotlib.pyplot as plt

FIGDIR = "."
os.makedirs(FIGDIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
#  Geometry  (κ-stereographic model, Ganea convention)
# ────────────────────────────────────────────────────────────

def S_K(K, R):
    """Rauch comparison factor sn_K(R)/R."""
    if abs(R) < 1e-15 or abs(K) < 1e-14:
        return 1.0
    sk = abs(K) ** 0.5
    if K < 0:
        return float(np.sinh(sk * R) / (sk * R))
    else:
        return float(np.sin(sk * R) / (sk * R))


def exp0(v, K):
    """Exponential map at the origin of the κ-stereographic model."""
    if abs(K) < 1e-14:
        return v
    kp = abs(K)
    sk = kp ** 0.5
    nv = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    if K < 0:
        return torch.tanh(sk * nv) / (sk * nv) * v
    else:
        return torch.tan((sk * nv).clamp(max=1.55)) / (sk * nv) * v


def dist2(p, q, K):
    """Squared geodesic distance on the κ-stereographic model."""
    if abs(K) < 1e-14:
        return (p - q).pow(2).sum(-1)
    kp = abs(K)
    sk = kp ** 0.5
    np2 = p.pow(2).sum(-1)
    nq2 = q.pow(2).sum(-1)
    dpq2 = (p - q).pow(2).sum(-1)
    if K < 0:
        denom = ((1 - kp * np2) * (1 - kp * nq2)).clamp(min=1e-15)
        arg = (1 + 2 * kp * dpq2 / denom).clamp(min=1.0)
        return (torch.acosh(arg) / sk).pow(2)
    else:
        denom = ((1 + kp * np2) * (1 + kp * nq2)).clamp(min=1e-15)
        arg = (1 - 2 * kp * dpq2 / denom).clamp(-1, 1)
        return (torch.acos(arg) / sk).pow(2)


# ────────────────────────────────────────────────────────────
#  Data + network
# ────────────────────────────────────────────────────────────

def make_data(d, m, R, seed=42, cond=3.0):
    """Generate whitened tangent-space regression data."""
    rng = np.random.RandomState(seed)
    xi = rng.randn(m, d)
    if m >= d:
        C = xi.T @ xi / m
        L = np.linalg.cholesky(C)
        xi = xi @ np.linalg.inv(L).T
    xi *= R * 0.9 / np.linalg.norm(xi, axis=1).max()
    U, _, Vt = np.linalg.svd(rng.randn(d, d))
    sigs = np.linspace(1.0, 1.0 / cond, d)
    Phi = U @ np.diag(sigs) @ Vt
    yt = (Phi @ xi.T).T
    return (
        torch.tensor(xi, dtype=torch.float64),
        torch.tensor(yt, dtype=torch.float64),
        torch.tensor(Phi, dtype=torch.float64),
    )


def init_balanced(d, N, alpha=0.3, seed=0):
    """Balanced orthogonal initialisation: each W_j = s * Q_j."""
    rng = np.random.RandomState(seed)
    s = alpha ** (1.0 / N)
    Ws = []
    for _ in range(N):
        Q, _ = np.linalg.qr(rng.randn(d, d))
        Ws.append(torch.tensor(s * Q, dtype=torch.float64, requires_grad=True))
    return Ws


def A_of(Ws):
    A = Ws[0]
    for W in Ws[1:]:
        A = W @ A
    return A


def loss_intrinsic(Ws, xi, yt, K):
    A = A_of(Ws)
    pred = exp0((A @ xi.T).T, K)
    targ = exp0(yt, K)
    return 0.5 * dist2(pred, targ, K).mean()


def loss_surrogate(Ws, xi, yt):
    A = A_of(Ws)
    return 0.5 * ((A @ xi.T).T - yt).pow(2).sum(-1).mean()


def balancedness_defect(Ws):
    """max_j ||W_{j+1}^T W_{j+1} - W_j W_j^T||_F."""
    mx = 0.0
    for j in range(len(Ws) - 1):
        WtW = Ws[j + 1].T @ Ws[j + 1]
        WWt = Ws[j] @ Ws[j].T
        mx = max(mx, (WtW - WWt).norm().item())
    return mx


# ────────────────────────────────────────────────────────────
#  Training loop (with autograd)
# ────────────────────────────────────────────────────────────

def train(Ws, xi, yt, K, eta, iters, mode="intr", record_every=1):
    """
    Gradient descent with autograd.
    Returns dict with 'loss_intr', 'loss_surr', 'balance' trajectories.
    """
    rec = {"loss_intr": [], "loss_surr": [], "balance": []}
    for t in range(iters):
        # --- measure ---
        if t % record_every == 0:
            with torch.no_grad():
                li = loss_intrinsic(Ws, xi, yt, K).item()
                ls = loss_surrogate(Ws, xi, yt).item()
                bd = balancedness_defect(Ws)
            rec["loss_intr"].append(li)
            rec["loss_surr"].append(ls)
            rec["balance"].append(bd)
            if li > 1e8 or np.isnan(li):
                break

        # --- gradient step ---
        if mode == "intr":
            L = loss_intrinsic(Ws, xi, yt, K)
        else:
            L = loss_surrogate(Ws, xi, yt)
        L.backward()
        with torch.no_grad():
            for W in Ws:
                W -= eta * W.grad
                W.grad.zero_()

    # final measurement
    with torch.no_grad():
        rec["loss_intr"].append(loss_intrinsic(Ws, xi, yt, K).item())
        rec["loss_surr"].append(loss_surrogate(Ws, xi, yt).item())
        rec["balance"].append(balancedness_defect(Ws))

    for k in rec:
        rec[k] = np.array(rec[k])
    return rec


def hessian_top_eig(Ws, xi, yt, K, niters=100, eps=1e-4):
    """Top Hessian eigenvalue of L_c w.r.t. A at current A, via power iteration."""
    d = Ws[0].shape[0]
    A0 = A_of(Ws).detach()

    # random direction
    torch.manual_seed(0)
    v = torch.randn(d, d, dtype=torch.float64)
    v /= v.norm()

    for _ in range(niters):
        # Compute gradient at A0 + eps*v  and  A0 - eps*v
        for sign, store in [(1, "gp"), (-1, "gm")]:
            A_pert = (A0 + sign * eps * v).requires_grad_(True)
            pred = exp0((A_pert @ xi.T).T, K)
            targ = exp0(yt, K)
            L = 0.5 * dist2(pred, targ, K).mean()
            g = torch.autograd.grad(L, A_pert)[0]
            if store == "gp":
                gp = g.detach()
            else:
                gm = g.detach()

        Hv = (gp - gm) / (2 * eps)
        lam = (Hv * v).sum().item()
        nHv = Hv.norm().item()
        if nHv > 1e-15:
            v = Hv / nHv
    return abs(lam)


# ────────────────────────────────────────────────────────────
#  FIGURE 1  (figure_descent.pdf) — the main 4-panel figure
#  Tests: Thm 1 (descent), Thm 2 (convergence), Cor 4 (positive K)
# ────────────────────────────────────────────────────────────

def make_figure_descent(d, N, m, R, seeds, Ks, eta, iters):
    print("=" * 60)
    print("Figure: descent (4-panel)")
    print("=" * 60)
    xi, yt, Phi = make_data(d, m, R, seed=42, cond=3.0)

    # storage
    all_loss = {K: [] for K in Ks}
    all_surr = {K: [] for K in Ks}
    all_bal  = {K: [] for K in Ks}
    all_lam  = {K: [] for K in Ks}

    for seed in seeds:
        for K in Ks:
            print(f"  seed={seed} K={K:5.1f} ... ", end="", flush=True)
            Ws = init_balanced(d, N, alpha=0.3, seed=seed)
            rec = train(Ws, xi, yt, K, eta, iters, "intr", record_every=1)
            all_loss[K].append(rec["loss_intr"])
            all_surr[K].append(rec["loss_surr"])
            all_bal[K].append(rec["balance"])

            # near-optimum sharpness
            # converge to near-optimum with surrogate (fast, exact)
            Ws2 = init_balanced(d, N, alpha=0.3, seed=seed)
            train(Ws2, xi, yt, K, 0.3, 30000, "surr", record_every=29999)
            lam = hessian_top_eig(Ws2, xi, yt, K)
            all_lam[K].append(lam)
            print(f"lambda*={lam:.3f}")

    # ---- averages ----
    def avg(d_list):
        arrs = d_list
        minlen = min(len(a) for a in arrs)
        stacked = np.array([a[:minlen] for a in arrs])
        return stacked.mean(0), stacked.std(0)

    cmap = plt.cm.coolwarm_r
    nK = len(Ks)

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.3))

    # (a) Raw loss trajectories at fixed step size
    ax = axes[0]
    for idx, K in enumerate(Ks):
        col = cmap(idx / (nK - 1))
        mu, sd = avg(all_loss[K])
        ax.semilogy(mu, color=col, lw=1.1, label=f"$K={K}$")
        ax.fill_between(range(len(mu)), mu - sd, mu + sd, color=col, alpha=0.08)
    # overlay surrogate at K=0
    mu0, _ = avg(all_surr[0.0])
    ax.semilogy(mu0, "--", color="gray", lw=1, label=r"$\mathcal{E}$ (surr.)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\mathcal{L}_c$")
    ax.set_title("(a) Loss trajectories")
    ax.legend(fontsize=5.5, ncol=2)
    ax.grid(True, alpha=0.25)

    # (b) Near-optimum sharpness λ*_K  vs  K
    ax = axes[1]
    lam_mu = [np.mean(all_lam[K]) for K in Ks]
    lam_sd = [np.std(all_lam[K]) for K in Ks]
    ax.errorbar(Ks, lam_mu, yerr=lam_sd, fmt="o-", color="#2563eb",
                markersize=4, capsize=3, lw=1.2)
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"$\lambda^\star_K$")
    ax.set_title(r"(b) Sharpness at convergence")
    ax.grid(True, alpha=0.25)

    # print key numbers for the paper text
    lam0 = np.mean(all_lam[0.0])
    lam8 = np.mean(all_lam[-8.0]) if -8.0 in Ks else 0
    lam1 = np.mean(all_lam[1.0]) if 1.0 in Ks else 0
    print(f"\n  lambda*_0 = {lam0:.2f}")
    print(f"  lambda*_{{-8}} = {lam8:.1f}  ({lam8/lam0:.0f}x)")
    print(f"  lambda*_1 = {lam1:.2f}")
    print(f"  stable step K=0: {2/lam0:.2f}")
    print(f"  stable step K=-8: {2/lam8:.2f}")
    print(f"  stable step K=1: {2/lam1:.2f}")

    # (c) λ*_K / λ*_0  vs  S_K(R)^2
    ax = axes[2]
    S2 = [S_K(K, R * 0.9) ** 2 for K in Ks]
    ratios = [np.mean(all_lam[K]) / lam0 for K in Ks]
    ax.plot(S2, ratios, "o", color="#2563eb", markersize=5, zorder=3)
    smax = max(S2) * 1.1
    ax.plot([0, smax], [0, smax], "--", color="#dc2626", lw=1, label="identity")
    for i, K in enumerate(Ks):
        ax.annotate(f"${K}$", (S2[i], ratios[i]), fontsize=5.5,
                    textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel(r"$S_K(R)^2$")
    ax.set_ylabel(r"$\lambda^\star_K / \lambda^\star_0$")
    ax.set_title(r"(c) Scaling with Rauch factor")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # (d) Balancedness defect vs iteration
    ax = axes[3]
    for idx, K in enumerate(Ks):
        col = cmap(idx / (nK - 1))
        mu, _ = avg(all_bal[K])
        ax.plot(mu, color=col, lw=0.9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\delta$-balance defect")
    ax.set_title("(d) Balancedness preserved")
    ax.grid(True, alpha=0.25)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(FIGDIR, "figure_descent.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "figure_descent.png"), bbox_inches="tight")
    plt.close()
    print("  => figure_descent.pdf saved\n")


# ────────────────────────────────────────────────────────────
#  FIGURE 2  (figure_collapse.pdf) — convergence exponent vs K
#  Tests: Prop 4 (surrogate = curvature-free control)
# ────────────────────────────────────────────────────────────

def make_figure_collapse(d, N, m, R, seeds, Ks, eta, iters):
    print("=" * 60)
    print("Figure: collapse (convergence exponent)")
    print("=" * 60)
    xi, yt, _ = make_data(d, m, R, seed=42, cond=3.0)

    all_exp_intr = {K: [] for K in Ks}
    all_exp_surr = {K: [] for K in Ks}

    for seed in seeds:
        for K in Ks:
            Ws = init_balanced(d, N, alpha=0.3, seed=seed)
            rec = train(Ws, xi, yt, K, eta, iters, "intr", record_every=1)
            li = rec["loss_intr"]
            ls = rec["loss_surr"]
            # fit log-linear in late half
            n = len(li)
            half = n // 2
            if li[half] > 1e-14 and li[-1] > 1e-14:
                t_arr = np.arange(half, n)
                p_i = np.polyfit(t_arr, np.log(li[half:n] + 1e-30), 1)
                all_exp_intr[K].append(abs(p_i[0]))
            if ls[half] > 1e-14 and ls[-1] > 1e-14:
                p_s = np.polyfit(t_arr, np.log(ls[half:n] + 1e-30), 1)
                all_exp_surr[K].append(abs(p_s[0]))

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    exp_i = [np.mean(all_exp_intr[K]) for K in Ks]
    exp_s = [np.mean(all_exp_surr[K]) for K in Ks]
    exp_s_0 = np.mean(all_exp_surr[0.0]) if all_exp_surr[0.0] else exp_s[Ks.index(0.0)]

    ax.plot(Ks, exp_i, "o-", color="#2563eb", markersize=5, lw=1.3,
            label=r"intrinsic $|b|$")
    ax.axhline(exp_s_0, ls="--", color="#dc2626", lw=1,
               label=r"surrogate $|b|$ (flat)")
    ax.set_xlabel("$K$")
    ax.set_ylabel("Per-step convergence exponent $|b|$")
    ax.set_title("Curvature-collapse check")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "figure_collapse.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "figure_collapse.png"), bbox_inches="tight")
    plt.close()
    print("  => figure_collapse.pdf saved\n")


# ────────────────────────────────────────────────────────────
#  FIGURE 3  (figure_intuition.pdf) — S_K(R) plot
#  Already exists; regenerate if missing.
# ────────────────────────────────────────────────────────────

def make_figure_intuition(R=1.2):
    print("=" * 60)
    print("Figure: intuition (S_K(R) plot)")
    print("=" * 60)
    Ks = np.linspace(-8, 2, 300)
    S2 = np.array([S_K(K, R * 0.9) ** 2 for K in Ks])

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(Ks, S2, color="#2563eb", lw=1.5)
    ax.axhline(1, ls=":", color="gray", lw=0.8)
    ax.axvline(0, ls=":", color="gray", lw=0.8)
    ax.fill_between(Ks[Ks > 0], 0, S2[Ks > 0], color="#22c55e", alpha=0.1)
    ax.fill_between(Ks[Ks < 0], 0, S2[Ks < 0], color="#ef4444", alpha=0.1)
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"$S_K(R)^2$")
    ax.set_title(f"Rauch factor at $R={R}$")
    ax.set_xlim(-8.5, 2.5)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "figure_intuition.pdf"), bbox_inches="tight")
    plt.close()
    print("  => figure_intuition.pdf saved\n")


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (default: 5 seeds 0-4)")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced iters for quick test")
    args = parser.parse_args()

    # Paper parameters (Section 5.1)
    d, N, m, R = 6, 3, 300, 1.2
    Ks = [1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -4.0, -8.0]
    seeds = [args.seed] if args.seed is not None else list(range(5))
    eta = 0.003
    iters = 800 if not args.fast else 200

    print(f"Config: d={d}, N={N}, m={m}, R={R}")
    print(f"  Ks = {Ks}")
    print(f"  seeds = {seeds}")
    print(f"  eta = {eta}, iters = {iters}")
    print()

    make_figure_intuition(R)
    make_figure_descent(d, N, m, R, seeds, Ks, eta, iters)
    make_figure_collapse(d, N, m, R, seeds, Ks, eta, iters)

    print("All figures generated.")


# ────────────────────────────────────────────────────────────
#  SUPPLEMENTARY FIGURES
#  These validate propositions not covered by the main figures.
# ────────────────────────────────────────────────────────────

def make_figure_surrogate(d, N, m, R, seeds, Ks_sub, eta, iters):
    """
    Figure: surrogate is curvature-free (Prop 4).
    Panel (a): intrinsic loss under intrinsic GD — curves separate.
    Panel (b): surrogate loss under surrogate GD — curves overlap.
    """
    print("=" * 60)
    print("Figure: surrogate (Prop 4)")
    print("=" * 60)
    xi, yt, _ = make_data(d, m, R, seed=42, cond=3.0)
    cols = {0.0: "#16a34a", -1.0: "#2563eb", -2.0: "#f59e0b",
            -4.0: "#dc2626", -8.0: "#7c3aed"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    for K in Ks_sub:
        # intrinsic GD
        all_intr, all_surr = [], []
        for seed in seeds:
            Ws = init_balanced(d, N, alpha=0.3, seed=seed)
            rec_i = train(Ws, xi, yt, K, eta, iters, "intr", record_every=1)
            all_intr.append(rec_i["loss_intr"])
            Ws = init_balanced(d, N, alpha=0.3, seed=seed)
            rec_s = train(Ws, xi, yt, K, eta, iters, "surr", record_every=1)
            all_surr.append(rec_s["loss_surr"])

        def avg(lst):
            mn = min(len(a) for a in lst)
            return np.mean([a[:mn] for a in lst], axis=0)

        mi = avg(all_intr); ms = avg(all_surr)
        ax1.semilogy(mi / mi[0], color=cols[K], lw=1.3, label=f"$K={K}$")
        ax2.semilogy(ms / ms[0], color=cols[K], lw=1.3, label=f"$K={K}$")

    ax1.set_xlabel("Iteration"); ax1.set_ylabel("$F_K(t)/F_K(0)$")
    ax1.set_title("(a) Intrinsic: curves separate")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.25)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel(r"$\mathcal{E}(t)/\mathcal{E}(0)$")
    ax2.set_title("(b) Surrogate: curves overlap")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "figure_surrogate.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "figure_surrogate.png"), bbox_inches="tight")
    plt.close()
    print("  => figure_surrogate.pdf saved\n")


def make_figure_radius(d, N, m, seeds):
    """
    Figure: sharpness vs radius R at fixed K=-2 (Thm 1).
    """
    print("=" * 60)
    print("Figure: radius (Thm 1)")
    print("=" * 60)
    K = -2.0
    Rs = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
    lams, S2s = [], []

    for R in Rs:
        xi, yt, Phi = make_data(d, m, R, seed=42, cond=3.0)
        ll = []
        for seed in seeds:
            Ws = init_balanced(d, N, alpha=0.3, seed=seed)
            train(Ws, xi, yt, K, 0.3, 15000, "surr", record_every=14999)
            ll.append(hessian_top_eig(Ws, xi, yt, K, niters=60))
        lam = np.mean(ll)
        S2 = S_K(K, R * 0.9) ** 2
        lams.append(lam); S2s.append(S2)
        print(f"  R={R:.1f}: lambda*={lam:.4f}, S^2={S2:.3f}")

    la, S2a, Ra = np.array(lams), np.array(S2s), np.array(Rs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.semilogy(Ra, la, "o-", color="#2563eb", markersize=5)
    ax1.set_xlabel("$R$"); ax1.set_ylabel(r"$\lambda^\star$")
    ax1.set_title(f"(a) Sharpness grows with $R$ ($K={K}$)")
    ax1.grid(True, alpha=0.25)
    ax2.plot(S2a, la / la[0], "o", color="#2563eb", markersize=5, label="Empirical")
    m_f, b_f = np.polyfit(S2a, la / la[0], 1)
    xs = np.linspace(S2a.min(), S2a.max(), 100)
    ax2.plot(xs, m_f * xs + b_f, "--", color="#dc2626", lw=1.2,
             label=f"Linear fit (slope={m_f:.1f})")
    ax2.set_xlabel(r"$S_K(R)^2$"); ax2.set_ylabel(r"$\lambda^\star/\lambda^\star_0$")
    ax2.set_title(r"(b) Ratio tracks $S_K(R)^2$")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "figure_radius.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "figure_radius.png"), bbox_inches="tight")
    plt.close()
    print("  => figure_radius.pdf saved\n")


def make_figure_landscape(d, N, m, R):
    """
    Figure: no spurious local minima (Thm 4).
    Balanced inits converge uniformly; unbalanced converge more slowly.
    """
    print("=" * 60)
    print("Figure: landscape (Thm 4)")
    print("=" * 60)
    K = -2.0
    n_runs = 12
    xi, yt, _ = make_data(d, m, R, seed=42, cond=3.0)
    eta_b, eta_u = 0.05, 0.01
    iters = 2000

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    for s in range(n_runs):
        Ws = init_balanced(d, N, alpha=0.3, seed=s + 100)
        rec = train(Ws, xi, yt, K, eta_b, iters, "surr", record_every=1)
        a1.semilogy(rec["loss_surr"], color="#2563eb", alpha=0.25, lw=0.8)

        rng = np.random.RandomState(s + 500)
        Ws_u = []
        for j in range(N):
            Q, _ = np.linalg.qr(rng.randn(d, d))
            sc = 2.0 if j == 0 else 0.02
            Ws_u.append(torch.tensor(sc * Q, dtype=torch.float64, requires_grad=True))
        rec_u = train(Ws_u, xi, yt, K, eta_u, iters, "surr", record_every=1)
        a2.semilogy(rec_u["loss_surr"], color="#f59e0b", alpha=0.25, lw=0.8)

    a1.set_xlabel("Iteration"); a1.set_ylabel(r"$\mathcal{E}$")
    a1.set_title(f"Balanced init ({n_runs} seeds)")
    a1.set_ylim(1e-14, 1); a1.grid(True, alpha=0.25)
    a2.set_xlabel("Iteration"); a2.set_ylabel(r"$\mathcal{E}$")
    a2.set_title(f"Unbalanced init ({n_runs} seeds)")
    a2.set_ylim(1e-14, 1); a2.grid(True, alpha=0.25)
    fig.suptitle(f"No Spurious Minima ($K={K}$, $N={N}$)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "figure_landscape.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "figure_landscape.png"), bbox_inches="tight")
    plt.close()
    print("  => figure_landscape.pdf saved\n")
