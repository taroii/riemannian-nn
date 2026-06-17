"""Known-curvature synthetic regression generator (plan section 5).

We sample inputs on a manifold of *known* curvature ``kappa_data`` so the
phase-diagram x-axis ``sqrt(|kappa_data|)`` is exact, and we hold the
tangent-lifted problem fixed across ``kappa_data`` (plan section 5: "lift a
single fixed problem through the manifold, so curvature enters only through the
geometry, not the data").

Construction
------------
1. Draw a fixed latent ``z`` in the tangent space at the origin, ``z ~ N(0, I)``
   scaled so its norm stays inside a geodesic ball of radius ``radius``.
2. A FIXED teacher (seeded independently of ``kappa_data``) maps ``z`` to a
   regression target ``t = teacher(z) + noise``. Because the teacher and the
   latent draw are seeded independently of curvature, the underlying problem is
   identical across the curvature sweep.
3. The model input is the manifold point ``x = expmap0_{kappa_data}(z)``.

IMPORTANT design caveat (read before trusting P1 empirically)
-------------------------------------------------------------
A model whose first operation is ``logmap0`` at the SAME curvature ``kappa_data``
recovers ``z`` exactly (``log_o . exp_o = id`` at the origin), so a *single*
matched-curvature layer is curvature-trivial. Curvature enters the *learned
function* empirically only through (a) hidden-manifold geometry and depth
(``kappa_model``, and hence ``Lambda_N`` -- prediction P2) for ``N >= 2`` layers,
where the composition ``exp_{kappa_model}(W log_{kappa_model}(.))`` does not
telescope to the identity; and (b) the analytic Bishop--Gromov input penalty
``sqrt(|kappa_data|)/Lambda_N`` that the bound adds (prediction P1, overlaid as
contours rather than learned). Therefore sweeps must use ``N >= 2``, and the
choice of how strongly ``kappa_data`` should drive the *measured* gap is an open
task-design decision (plan section 11) -- revisit it on the server before the
full run. The generator below is correct and runnable; it does not pre-judge
that decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rnn import manifolds


@dataclass
class SyntheticConfig:
    kappa_data: float = -1.0          # signed input-manifold curvature
    d0: int = 8                       # input manifold dimension
    d_out: int = 1                    # regression target dimension
    n_train: int = 512
    n_test: int = 2048                # large test set -> low-variance gap estimate
    radius: float = 1.0               # geodesic-ball radius the latents live in
    teacher_hidden: int = 32          # fixed teacher MLP width
    noise_std: float = 0.05
    teacher_seed: int = 1234          # FIXED across the curvature sweep
    dtype: torch.dtype = manifolds.DEFAULT_DTYPE


def _fixed_teacher(z: torch.Tensor, cfg: SyntheticConfig) -> torch.Tensor:
    """A small fixed nonlinear teacher t = W2 tanh(W1 z), seeded independent of kappa."""
    g = torch.Generator().manual_seed(cfg.teacher_seed)
    d0, h, do = cfg.d0, cfg.teacher_hidden, cfg.d_out
    W1 = torch.randn(d0, h, generator=g, dtype=cfg.dtype) / (d0 ** 0.5)
    W2 = torch.randn(h, do, generator=g, dtype=cfg.dtype) / (h ** 0.5)
    return torch.tanh(z @ W1) @ W2


def make_synthetic_regression(cfg: SyntheticConfig, seed: int):
    """Return ``(x_train, t_train, x_test, t_test, meta)``.

    ``x_*`` are points on the kappa_data-Stereographic manifold (ambient
    coordinates, shape ``[n, d0]``); ``t_*`` are regression targets ``[n, d_out]``.
    ``meta`` carries the realized latent radius (used to size the geometric tube
    for instrumentation) and the curvature.
    """
    manifold = manifolds.stereographic(cfg.kappa_data)
    g = torch.Generator().manual_seed(seed)

    def _sample(n: int):
        z = torch.randn(n, cfg.d0, generator=g, dtype=cfg.dtype)
        # Rescale each latent to a geodesic radius uniform in (0, radius] so the
        # tangent norms (geodesic distances from the origin) are controlled.
        u = torch.rand(n, 1, generator=g, dtype=cfg.dtype)
        z = torch.nn.functional.normalize(z, dim=1) * (u * cfg.radius)
        t = _fixed_teacher(z, cfg) + cfg.noise_std * torch.randn(
            n, cfg.d_out, generator=g, dtype=cfg.dtype
        )
        x = manifold.expmap0(z)
        return x, t

    x_train, t_train = _sample(cfg.n_train)
    x_test, t_test = _sample(cfg.n_test)
    meta = {
        "kappa_data": cfg.kappa_data,
        "sqrt_abs_kappa_data": manifolds.sqrt_abs_kappa(cfg.kappa_data),
        "d0": cfg.d0,
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "input_tube_radius": cfg.radius,
    }
    return x_train, t_train, x_test, t_test, meta
