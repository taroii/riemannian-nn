"""Evaluation of the curvature-adaptive bound (paper eq. 16, plan section 2).

The bound on the generalization gap ``Delta = |R(theta) - R_hat_n(theta)|`` is

    Delta  <~  [ L_l sqrt( d0 log(L_l Lambda_N sqrt(n)) ) + sqrt(|kappa_data|)/Lambda_N ]
                 / n^(1/d0)
             +  B sqrt( log(1/delta) / n ).

CRITICAL (plan section 2): evaluate the LITERAL formula at each model's MEASURED
``Lambda_N`` -- never a hand-simplified/monotone version. The two curvature
channels move in opposite directions (the covering term rises in Lambda_N, the
data-curvature term falls), so the net dependence need not be monotone. As
``kappa_data -> 0`` the ``sqrt(|kappa|)/Lambda_N`` term vanishes and the bound
reduces to the Euclidean covering bound (the recovery check).

We validate the *shape*, not the magnitude (plan section 6): the headline fits a
single global multiplicative constant; the appendix reports Kendall-tau rank
correlation. So we expose:
  - :func:`evaluate_bound`    -- the full literal value (for the rank-correlation table).
  - :func:`bound_predictor`   -- the curvature/Lambda_N-dependent scalar predictor
    (the collapse-plot x-axis and the quantity the global constant scales).
"""

from __future__ import annotations

import math


def _covering_term(L_l: float, d0: int, Lambda_N: float, n: int) -> float:
    """L_l sqrt( d0 log(L_l Lambda_N sqrt(n)) ) -- the model-side covering term."""
    arg = L_l * Lambda_N * math.sqrt(n)
    # log argument must exceed 1 for a real, non-negative covering term; clamp so a
    # tiny Lambda_N (e.g. the kappa->0 Euclidean limit) cannot produce NaNs.
    log_arg = math.log(max(arg, 1.0 + 1e-12))
    return L_l * math.sqrt(max(d0 * log_arg, 0.0))


def bound_predictor(
    *, L_l: float, d0: int, Lambda_N: float, n: int, kappa_data: float
) -> float:
    """The curvature/Lambda_N-dependent part of eq. 16 (the collapse predictor).

    This is the bracketed numerator divided by the rate ``n^(1/d0)``; it excludes
    only the config-independent confidence term ``B sqrt(log(1/delta)/n)``.
    """
    covering = _covering_term(L_l, d0, Lambda_N, n)
    data_curv = math.sqrt(abs(kappa_data)) / max(Lambda_N, _SMALL)
    rate = n ** (1.0 / d0)
    return (covering + data_curv) / rate


def evaluate_bound(
    *,
    L_l: float,
    d0: int,
    Lambda_N: float,
    n: int,
    kappa_data: float,
    B: float,
    delta: float = 0.05,
) -> float:
    """The full literal bound value of eq. 16 (for the Kendall-tau table)."""
    predictor = bound_predictor(
        L_l=L_l, d0=d0, Lambda_N=Lambda_N, n=n, kappa_data=kappa_data
    )
    confidence = B * math.sqrt(math.log(1.0 / delta) / n)
    return predictor + confidence


_SMALL = 1e-12
