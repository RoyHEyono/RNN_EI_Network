"""Generate JSON configs for SLURM array jobs: per lr, ei + vanilla/LSTM (± LayerNorm), same seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Single seed for all grid runs; learning rates sweep below.
GRID_SEED = 42

# Log-spaced steps between anchors 0.001 → 0.01 → 0.1 → 0.5 (×2, ×2.5, ×2 between decades).
LEARNING_RATES: list[float] = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "NeuroGym batch_run config generator. Grid over LEARNING_RATES with fixed "
            f"seed {GRID_SEED}; per lr: ei, ei+param-ln, vanilla±LN, lstm±LN — 6×len(LR) rows."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "random_configs.json",
        help="path to output JSON file",
    )
    return p


def grid_configs() -> list[dict[str, float | int | str | bool | None]]:
    configs: list[dict[str, float | int | str | bool | None]] = []
    for lr in LEARNING_RATES:
        # EI: no --layer-norm (N/A); vanilla/lstm use the flag from JSON.
        configs.append({"lr": lr, "seed": GRID_SEED, "arch": "ei", "layer_norm": True})
        # EI with ParametrizedLayerNorm (param-ln specific hparams left at CLI defaults).
        configs.append(
            {"lr": lr, "seed": GRID_SEED, "arch": "ei", "param_layer_norm": True}
        )
        configs.append(
            {"lr": lr, "seed": GRID_SEED, "arch": "vanilla", "layer_norm": True}
        )
        configs.append(
            {"lr": lr, "seed": GRID_SEED, "arch": "vanilla", "layer_norm": False}
        )
        configs.append(
            {"lr": lr, "seed": GRID_SEED, "arch": "lstm", "layer_norm": True}
        )
        configs.append(
            {"lr": lr, "seed": GRID_SEED, "arch": "lstm", "layer_norm": False}
        )
    return configs


def main() -> None:
    args = build_parser().parse_args()
    configs = grid_configs()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    n = len(configs)
    n_lr = len(LEARNING_RATES)
    print(f"Wrote {n} configs ({n_lr} LRs × 6: ei, ei+param-ln, vanilla±LN, lstm±LN) to {args.out}")
    print(f"SLURM: #SBATCH --array=0-{n - 1}")


if __name__ == "__main__":
    main()
