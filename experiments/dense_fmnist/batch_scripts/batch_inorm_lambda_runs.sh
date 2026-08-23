#!/usr/bin/env bash
#SBATCH --array=0-19%2      # 4 luminosities x 5 lambda values
#SBATCH --partition=long
#SBATCH --exclude=cn-c008
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"  # skip volta (V100, CC 7.0): unsupported by our torch build
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_lambda_%A_%a.out
#SBATCH --error=sbatch_err/grid_lambda_%A_%a.err
#SBATCH --job-name=grid_lambda
#
# Sensitivity of I-Norm to the weight of the normalization loss (the default
# used everywhere else is 0.01).
set -euo pipefail

# sbatch runs a spooled copy; fall back to the submit directory for siblings.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || true
if [[ ! -f "${_SCRIPT_DIR}/_common.sh" ]]; then
  _SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
source "${_SCRIPT_DIR}/_common.sh"

brightness_factors=(0 0.25 0.5 0.75)
lambdas=(0.00001 0.0001 0.001 0.1 1)

i=${SLURM_ARRAY_TASK_ID}
bf=${brightness_factors[$(( i % 4 ))]}
lam=${lambdas[$(( (i / 4) % 5 ))]}

echo "grid=$i brightness=$bf lambda=$lam"

export BRIGHTNESS_FACTOR=$bf
export NORMTYPE_DETACH=1
export LN_FEEDBACK="full"
export SHUNTING=1
export LAMBDA_HOMEOS=$lam

submit_inner_array "$SCRIPT_DIR/run_inorm_network.sh"
