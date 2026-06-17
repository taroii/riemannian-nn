"""Curvature conventions and geoopt manifold construction.

ONE convention, end to end (plan sections 0 and 9 -- the bookkeeping warning):

    ``kappa`` is the *signed sectional curvature* of the kappa-Stereographic
    model, matching geoopt's ``k`` argument exactly:

        kappa < 0   hyperbolic (Poincare ball of curvature kappa)
        kappa = 0   Euclidean
        kappa > 0   spherical (stereographic sphere)

The paper writes its penalties in terms of the curvature *magnitude*
``|kappa|`` and ``sqrt(|kappa|)`` (Bishop--Gromov). Use :func:`sqrt_abs_kappa`
everywhere the bound needs ``sqrt(|kappa|)`` so the ``kappa`` -> ``sqrt(|kappa|)``
map is in exactly one place.

Numerics caveat (plan section 0): the Poincare/stereographic model is unstable
in float32. Build geometry in float64 -- see :data:`DEFAULT_DTYPE`.
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

    This is the single sweep knob of plan section 3: changing ``kappa`` moves the
    geometry continuously across hyperbolic / Euclidean / spherical regimes. At
    ``kappa == 0`` geoopt returns the Euclidean limit, which is exactly the
    ``kappa_data -> 0`` recovery check (plan section 2 / section 9).
    """
    return geoopt.Stereographic(k=kappa, learnable=learnable)


def s_kappa(kappa: float, radius: float) -> float:
    """The angular-distortion factor ``S_kappa(R)`` from the paper.

    For hyperbolic curvature (kappa < 0) with magnitude ``c = |kappa|``,

        S_kappa(R) = sinh(sqrt(c) R) / (sqrt(c) R),

    and ``S_kappa(R) -> 1`` as ``kappa -> 0``. This is the analytic upper bound on
    ``alpha_j`` quoted in the paper; the instrumentation
    (:mod:`rnn.instrument.lambda_n`) measures the realized constant instead, but
    this closed form is useful as a sanity check and for the Euclidean-recovery
    test.
    """
    c = abs(kappa)
    if c == 0.0 or radius == 0.0:
        return 1.0
    x = math.sqrt(c) * radius
    if kappa < 0:  # hyperbolic
        return math.sinh(x) / x
    return math.sin(x) / x  # spherical


def to_dtype(x: torch.Tensor, dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    return x.to(dtype)
