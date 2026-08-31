#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONFIG_GLOB="${1:-experiments/rnn_fmnist/sweeps/generated/*.yaml}"

cd "$REPO_ROOT"
python -m experiments.rnn_fmnist.sweeps.generate_configs

shopt -s nullglob
configs=( $CONFIG_GLOB )
if (( ${#configs[@]} == 0 )); then
  echo "No sweep configs matched: $CONFIG_GLOB" >&2
  exit 2
fi

for config in "${configs[@]}"; do
  echo "Creating sweep from $config"
  uv run wandb sweep "$config"
done
