"""Curvature conventions and geoopt manifold construction.

ONE convention, end to end (paper §3 Preliminaries, "Model Spaces and Curvature
Conventions"):

    ``kappa`` is the *signed sectional curvature* of the kappa-Stereographic
    model, matching geoopt's ``k`` argument exactly:

        kappa < 0   hyperbolic (Poincare ball of curvature kappa)
        kappa = 0   Euclidean
        kappa > 0   spherical (stereographic sphere)

The paper writes its penalties in terms of the curvature *magnitude*
``|kappa|`` and ``sqrt(|kappa|)`` (§3: it sets kappa := |K| and the Rauch factor
S_K(R) = sn_K(R)/R). Use :func:`sqrt_abs_kappa` everywhere the bound needs
``sqrt(|kappa|)`` so the ``kappa`` -> ``sqrt(|kappa|)`` map is in exactly one place.

Numerics caveat: the Poincare/stereographic model is unstable in float32. Build
geometry in float64 -- see :data:`DEFAULT_DTYPE`.
"""

from __future__ import annotations

import math

import geoopt
import torch

DEFAULT_DTYPE = torch.float64


def sqrt_abs_kappa(kappa: float) -> float:
    """``sqrt(|kappa|)`` -- the quantity that appears in the bound's penalties."""
    return math.sqrt(abs(kappa))


def stereographic(kappa: float, *, learnable: bool = False) -> geoopt.Stereographic:
    """Build the kappa-Stereographic manifold of (signed) curvature ``kappa``.

    This is the single sweep knob of the experiments: changing ``kappa`` moves the
    geometry continuously across hyperbolic / Euclidean / spherical regimes (the
    single-chart property of the kappa-stereographic model, paper §3 / Appendix B).
    At ``kappa == 0`` geoopt returns the Euclidean limit, where the intrinsic loss
    coincides exactly with the tangent-space surrogate (paper §3, "Two Losses").
    """
    return geoopt.Stereographic(k=kappa, learnable=learnable)
