"""Lambda_N instrumentation (plan section 7) -- the genuinely new piece.

We extract the layerwise operator-norm constants

    alpha_j = sup ||D exp_{o_j}||        (expansion of the exp map)
    beta_j  = sup ||D log_{o_j}||        (expansion of the log map; beta_0 carries kappa_data)
    P_j     = sup ||D_u F_j||            (tangent-map / weight operator norm)

via matrix-free power iteration on each layer's Jacobian (autograd jvp/vjp),
maxed over the data batch (the sup over the geometric tube of plan section 7).
The network spatial Lipschitz constant is the product over layers

    Lambda_N = prod_{j=1}^N alpha_j P_j beta_{j-1}.

No existing repo computes this product; it is the theorem's object (plan section 0).
"""

from __future__ import annotations

import torch
from torch.autograd.functional import jvp, vjp

_EPS = 1e-30


@torch.no_grad()
def _normalize_rows(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=1, keepdim=True).clamp_min(_EPS)


def batched_op_norm(func, x: torch.Tensor, n_iter: int = 25, seed: int = 0) -> float:
    """Max over the batch of the per-sample Jacobian spectral norm of ``func`` at ``x``.

    ``func`` maps ``[B, d_in] -> [B, d_out]`` row-independently (true for expmap0,
    logmap0, and a Linear tangent op), so the Jacobian is block-diagonal over the
    batch and a single vectorized power iteration recovers every per-sample
    largest singular value at once. We return the max -- the ``sup`` over the tube.
    """
    x = x.detach()
    B = x.shape[0]
    g = torch.Generator(device=x.device).manual_seed(seed)
    v = torch.randn(x.shape, generator=g, dtype=x.dtype, device=x.device)
    v = _normalize_rows(v)
    for _ in range(n_iter):
        # J v   (forward-mode via autograd.functional)
        Jv = jvp(func, x, v)[1]
        # J^T (J v)
        JtJv = vjp(func, x, Jv)[1]
        v = _normalize_rows(JtJv)
    # Rayleigh quotient with the converged unit vector: ||J^T J v|| = sigma_max^2.
    Jv = jvp(func, x, v)[1]
    JtJv = vjp(func, x, Jv)[1]
    sigma = JtJv.norm(dim=1).clamp_min(0.0).sqrt()  # per-sample largest singular value
    return float(sigma.max().item())


def instrument_lambda_n(model, x: torch.Tensor, max_points: int = 256, n_iter: int = 25):
    """Measure (alpha_j, beta_j, P_j) per layer and aggregate Lambda_N.

    ``model`` is a :class:`rnn.models.LayerwiseRiemannianNN`; ``x`` a batch of
    input-manifold points. Returns a dict with per-layer lists (paper indexing,
    layers 1..N) and the scalar ``Lambda_N``.
    """
    if x.shape[0] > max_points:
        x = x[:max_points]
    N = model.cfg.depth
    _, trace = model(x, return_trace=True)

    alphas, betas, Ps = [], [], []
    for jp in range(N):
        m_in = trace["manifold_in"][jp]
        h_in = trace["h_in"][jp]
        a = trace["a"][jp]

        # beta_{j-1}: expansion of the input log map of this layer.
        beta = batched_op_norm(m_in.logmap0, h_in, n_iter=n_iter)
        # P_j: expansion of the tangent op F_j (Linear + activation) at its input u.
        u = m_in.logmap0(h_in).detach()
        P = batched_op_norm(lambda uu, _j=jp: model.tangent_op(_j, uu), u, n_iter=n_iter)
        # alpha_j: expansion of the exp map onto M_j (identity on the final layer).
        if jp < N - 1:
            alpha = batched_op_norm(model.m_hidden.expmap0, a, n_iter=n_iter)
        else:
            alpha = 1.0  # tangent-space (Euclidean) head: no exp on the last layer

        alphas.append(alpha)
        betas.append(beta)
        Ps.append(P)

    lambda_n = 1.0
    for jp in range(N):
        lambda_n *= alphas[jp] * Ps[jp] * betas[jp]

    return {
        "alpha": alphas,   # alpha_1 .. alpha_N (paper), alpha_N == 1
        "beta": betas,     # beta_0 .. beta_{N-1} (paper); beta_0 carries kappa_data
        "P": Ps,           # P_1 .. P_N (paper)
        "Lambda_N": float(lambda_n),
    }
