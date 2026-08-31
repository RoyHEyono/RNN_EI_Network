#!/usr/bin/env bash
#SBATCH --array=0-49%10
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/rnn_fmnist_sweep_%A_%a.out
#SBATCH --error=sbatch_err/rnn_fmnist_sweep_%A_%a.err
#SBATCH --job-name=rnn-fmnist-sweep

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mila/r/roy.eyono/RNN_EI_Network}"
SWEEP_ID="${SWEEP_ID:-${1:-}}"

if [[ -z "$SWEEP_ID" ]]; then
  echo "Usage: sbatch --export=ALL,SWEEP_ID=<entity/project/sweep-id> $0" >&2
  exit 2
fi

cd "$REPO_ROOT"
uv run wandb agent --count 1 "$SWEEP_ID"
