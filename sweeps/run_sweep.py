"""Config-driven grid runner (plan sections 7, 10).

Reads a YAML config describing the grid over (kappa_data, kappa_model, depth,
width, seed), trains one model per cell x seed, instruments Lambda_N, and writes
one row per (config, seed) to a parquet file under ``results/`` so analysis is
re-runnable without retraining.

    python -m sweeps.run_sweep --config sweeps/configs/smoke.yaml
    python sweeps/run_sweep.py   --config sweeps/configs/headline.yaml

The sweep is embarrassingly parallel across cells x seeds (plan section 12); this
driver runs serially for simplicity. Parallelize with a job array on the server
(each task a disjoint slice of the grid -> a parquet shard), then concatenate.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import pandas as pd
import yaml

# Allow `python sweeps/run_sweep.py` (no install) by adding the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn.data.synthetic import SyntheticConfig  # noqa: E402
from rnn.train import TrainConfig, run_single  # noqa: E402


def _grid(cfg: dict):
    g = cfg["grid"]
    keys = ["kappa_data", "kappa_model", "depth", "width", "seed"]
    axes = [g[k] for k in keys]
    for combo in itertools.product(*axes):
        yield dict(zip(keys, combo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="override output parquet path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fixed = cfg.get("fixed", {})
    tcfg_d = cfg.get("train", {})
    model_d = cfg.get("model", {})
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        cfg.get("name", "sweep") + ".parquet",
    )

    cells = list(_grid(cfg))
    print(f"[sweep] {cfg.get('name')}: {len(cells)} runs -> {out_path}")

    rows, t0 = [], time.time()
    for i, cell in enumerate(cells):
        data_cfg = SyntheticConfig(
            kappa_data=cell["kappa_data"],
            d0=fixed.get("d0", 8),
            d_out=fixed.get("d_out", 1),
            n_train=fixed.get("n_train", 512),
            n_test=fixed.get("n_test", 2048),
            radius=fixed.get("radius", 1.0),
        )
        train_cfg = TrainConfig(
            epochs=tcfg_d.get("epochs", 300),
            lr=tcfg_d.get("lr", 1e-2),
            weight_decay=tcfg_d.get("weight_decay", 0.0),
            matched_train_loss=tcfg_d.get("matched_train_loss", None),
        )
        row = run_single(
            kappa_data=cell["kappa_data"],
            kappa_model=cell["kappa_model"],
            depth=cell["depth"],
            width=cell["width"],
            d0=fixed.get("d0", 8),
            d_out=fixed.get("d_out", 1),
            seed=cell["seed"],
            mobius_bias=model_d.get("mobius_bias", False),
            mobius_bias_scale=model_d.get("mobius_bias_scale", 1e-3),
            data_cfg=data_cfg,
            train_cfg=train_cfg,
        )
        rows.append(row)
        if (i + 1) % max(1, len(cells) // 20) == 0 or i + 1 == len(cells):
            dt = time.time() - t0
            print(f"[sweep] {i + 1}/{len(cells)}  ({dt:.1f}s, gap={row['gap']:.4f}, "
                  f"Lambda_N={row['Lambda_N']:.3g})")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[sweep] wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
