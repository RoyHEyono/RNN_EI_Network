#!/usr/bin/env bash
#SBATCH --array=0-3       # 4 luminosities
#SBATCH --partition=long
#SBATCH --exclude=cn-c008
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"  # skip volta (V100, CC 7.0): unsupported by our torch build
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_inorm_div_%A_%a.out
#SBATCH --error=sbatch_err/grid_inorm_div_%A_%a.err
#SBATCH --job-name=grid_inorm_div
#
# Divisive-only control matched to batch_inorm_subtractive_runs.sh: the same
# local objective and sweep grid, but with the subtractive inhibitory pathway
# removed and the divisive pathway enabled.
set -euo pipefail

# sbatch runs a spooled copy; fall back to the submit directory for siblings.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || true
if [[ ! -f "${_SCRIPT_DIR}/_common.sh" ]]; then
  _SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
source "${_SCRIPT_DIR}/_common.sh"

brightness_factors=(0 0.25 0.5 0.75)
bf=${brightness_factors[${SLURM_ARRAY_TASK_ID}]}

echo "grid=${SLURM_ARRAY_TASK_ID} brightness=$bf shunting=1 subtractive=0"

export DATASET="${DATASET:-fashionmnist_contrast}"
export BRIGHTNESS_FACTOR=$bf
export NORMTYPE_DETACH=1
export LN_FEEDBACK="full"
export SHUNTING=1
export SUBTRACTIVE=0
export LAMBDA_HOMEOS=0.01

submit_inner_array "$SCRIPT_DIR/run_inorm_network.sh"
