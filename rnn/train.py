"""Single-run training + measurement (plan section 7).

One call trains one model on one synthetic dataset and returns a flat result row:
every knob, ``train_loss``, ``test_loss``, the gap ``Delta``, all per-layer
``alpha_j/beta_j/P_j``, ``Lambda_N``, the loss constants ``L_l/B``, and the
predicted bound. The sweep driver (``sweeps/run_sweep.py``) calls this across the
grid and persists the rows to parquet.

Gap protocol (plan sections 7, 9): the validated quantity is
``Delta = test_loss - train_loss`` (NEVER test error -- that conflates fit). We
also support early-stopping at a matched target train loss so fit is held roughly
constant across curvatures and only the gap varies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from rnn.data.synthetic import SyntheticConfig, make_synthetic_regression
from rnn.instrument.bound import bound_predictor, evaluate_bound
from rnn.instrument.lambda_n import instrument_lambda_n
from rnn.models.layerwise import LayerwiseRiemannianNN, ModelConfig


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-2
    weight_decay: float = 0.0
    matched_train_loss: float | None = None  # if set, early-stop at this train loss
    log_every: int = 0                        # 0 = silent


def _mse(pred, target):
    return F.mse_loss(pred, target)


def _loss_constants(model, x, t):
    """Measure L_l (loss Lipschitz, max per-sample grad norm) and B (loss bound)."""
    x = x.detach().clone().requires_grad_(False)
    pred = model(x)
    per_sample = ((pred - t) ** 2).mean(dim=1)        # [B]
    B = float(per_sample.max().item())
    # d/dpred of (pred-t)^2/d_out has norm proportional to ||pred - t||.
    L_l = float((pred - t).norm(dim=1).max().item())
    return L_l, B


def run_single(
    *,
    kappa_data: float,
    kappa_model: float,
    depth: int,
    width: int = 32,
    d0: int = 8,
    d_out: int = 1,
    seed: int = 0,
    mobius_bias: bool = False,
    mobius_bias_scale: float = 1e-3,
    data_cfg: SyntheticConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> dict:
    """Train one model and return a flat result row (a plain dict).

    ``mobius_bias`` adds an on-manifold bias per hidden layer; without it the
    network telescopes to a Euclidean MLP and curvature is cosmetic (see README
    "Key experimental finding"). Set it True to let kappa_model enter the function.
    """
    torch.manual_seed(seed)
    train_cfg = train_cfg or TrainConfig()

    if data_cfg is None:
        data_cfg = SyntheticConfig(kappa_data=kappa_data, d0=d0, d_out=d_out)
    else:
        data_cfg.kappa_data, data_cfg.d0, data_cfg.d_out = kappa_data, d0, d_out

    x_tr, t_tr, x_te, t_te, meta = make_synthetic_regression(data_cfg, seed=seed)

    model = LayerwiseRiemannianNN(
        ModelConfig(
            d0=d0, width=width, depth=depth, d_out=d_out,
            kappa_data=kappa_data, kappa_model=kappa_model,
            mobius_bias=mobius_bias, mobius_bias_scale=mobius_bias_scale,
        )
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    final_train_loss = float("nan")
    for epoch in range(train_cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = _mse(model(x_tr), t_tr)
        loss.backward()
        opt.step()
        final_train_loss = float(loss.item())
        if train_cfg.log_every and epoch % train_cfg.log_every == 0:
            print(f"  epoch {epoch:4d}  train_loss {final_train_loss:.5f}")
        if (
            train_cfg.matched_train_loss is not None
            and final_train_loss <= train_cfg.matched_train_loss
        ):
            break

    model.eval()
    with torch.no_grad():
        train_loss = float(_mse(model(x_tr), t_tr).item())
        test_loss = float(_mse(model(x_te), t_te).item())
    gap = test_loss - train_loss

    # Geometry instrumentation on the input distribution.
    instr = instrument_lambda_n(model, x_tr)
    L_l, B = _loss_constants(model, x_te, t_te)

    n = data_cfg.n_train
    predictor = bound_predictor(
        L_l=L_l, d0=d0, Lambda_N=instr["Lambda_N"], n=n, kappa_data=kappa_data
    )
    bound = evaluate_bound(
        L_l=L_l, d0=d0, Lambda_N=instr["Lambda_N"], n=n, kappa_data=kappa_data, B=B
    )

    row = {
        "kappa_data": kappa_data,
        "sqrt_abs_kappa_data": meta["sqrt_abs_kappa_data"],
        "kappa_model": kappa_model,
        "mobius_bias": mobius_bias,
        "depth": depth,
        "width": width,
        "d0": d0,
        "d_out": d_out,
        "seed": seed,
        "n_train": n,
        "n_test": data_cfg.n_test,
        "epochs_run": epoch + 1,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "gap": gap,
        "Lambda_N": instr["Lambda_N"],
        "L_l": L_l,
        "B": B,
        "bound_predictor": predictor,
        "bound_full": bound,
    }
    # Flatten per-layer constants for the appendix / aggregator tests.
    for jp, (al, be, pp) in enumerate(zip(instr["alpha"], instr["beta"], instr["P"])):
        row[f"alpha_{jp}"] = al
        row[f"beta_{jp}"] = be
        row[f"P_{jp}"] = pp
    return row
