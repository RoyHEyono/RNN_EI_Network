"""Generate the random hyperparameter configurations shared by every sweep arm.

Each sweep is a grid over experimental conditions crossed with the *same* list
of random configs, so runs can be paired across conditions by hyperparameter.
Ranges follow the paper's Table 1 as implemented in the original repo.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-configs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="the original script was unseeded; seeding makes sweeps reproducible")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "random_configs.json")
    return p


def sample_configs(num_configs: int, seed: int):
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(num_configs):
        configs.append(
            {
                "lr": float(10 ** rng.uniform(-3, -2)),        # excitatory
                "lr_wei": float(10 ** rng.uniform(-5, -2)),    # I -> E
                "lr_wix": float(10 ** rng.uniform(-2, 0)),     # E -> I
                "hidden_layer_width": int(rng.uniform(100, 500)),
            }
        )
    return configs


def main():
    args = build_parser().parse_args()
    if args.num_configs <= 0:
        raise ValueError("--num-configs must be > 0")
    configs = sample_configs(args.num_configs, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(configs, indent=2))
    print(f"Wrote {len(configs)} configs to {args.out}")


if __name__ == "__main__":
    main()
