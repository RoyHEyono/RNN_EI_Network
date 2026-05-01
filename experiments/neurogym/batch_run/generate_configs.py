"""Generate JSON configs for SLURM array jobs: paired ei vs vanilla with identical lr/seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "NeuroGym batch_run config generator. Samples num_pairs (lr, seed) draws and emits "
            "2×num_pairs rows: each draw twice with arch ei then vanilla (same lr and seed)."
        )
    )
    p.add_argument(
        "--num-pairs",
        type=int,
        default=20,
        metavar="N",
        help="number of shared (lr, seed) draws; JSON length will be 2N (match SBATCH --array=0-(2N-1))",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible config generation")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "random_configs.json",
        help="path to output JSON file",
    )
    return p


def sample_configs(num_pairs: int, seed: int) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    log_lr_lo, log_lr_hi = -2.52, -1.52
    configs: list[dict[str, float | int | str]] = []
    for _ in range(num_pairs):
        lr = float(10 ** rng.uniform(log_lr_lo, log_lr_hi))
        run_seed = int(rng.integers(0, 1_000_000_000))
        configs.append({"lr": lr, "seed": run_seed, "arch": "ei"})
        configs.append({"lr": lr, "seed": run_seed, "arch": "vanilla"})
    return configs


def main() -> None:
    args = build_parser().parse_args()
    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be > 0")
    configs = sample_configs(num_pairs=args.num_pairs, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    n = len(configs)
    print(f"Wrote {n} configs ({args.num_pairs} paired draws) to {args.out}")
    print(f"SLURM: #SBATCH --array=0-{n - 1}")


if __name__ == "__main__":
    main()
