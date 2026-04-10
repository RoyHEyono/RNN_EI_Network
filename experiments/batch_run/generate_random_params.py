import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate random batch-run configs for experiments.main."
    )
    parser.add_argument(
        "--num-configs",
        type=int,
        default=20,
        help="number of random configs to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible config generation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "random_configs.json",
        help="path to output JSON file",
    )
    return parser


def sample_configs(num_configs: int, seed: int):
    # Log-scale ranges for LR sweeps.
    lr_min, lr_max = -3, -2  # 1e-3 to 1e-2
    lr_ie_min, lr_ie_max = -5, -2  # 1e-5 to 1e-2
    lr_ei_min, lr_ei_max = -2, 0  # 1e-2 to 1e0

    rng = np.random.default_rng(seed)
    random_configs = []
    for _ in range(num_configs):
        random_configs.append(
            {
                "lr": float(10 ** rng.uniform(lr_min, lr_max)),
                "lr_ie": float(10 ** rng.uniform(lr_ie_min, lr_ie_max)),
                "lr_ei": float(10 ** rng.uniform(lr_ei_min, lr_ei_max)),
            }
        )
    return random_configs


def main():
    args = build_parser().parse_args()
    if args.num_configs <= 0:
        raise ValueError("--num-configs must be > 0")

    configs = sample_configs(num_configs=args.num_configs, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    print(f"Wrote {len(configs)} configs to {args.out}")


if __name__ == "__main__":
    main()