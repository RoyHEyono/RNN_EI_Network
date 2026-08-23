#!/usr/bin/env bash
#SBATCH --array=0-3%2       # 4 luminosities
#SBATCH --partition=long
#SBATCH --exclude=cn-c008
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"  # skip volta (V100, CC 7.0): unsupported by our torch build
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_inorm_sub_%A_%a.out
#SBATCH --error=sbatch_err/grid_inorm_sub_%A_%a.err
#SBATCH --job-name=grid_inorm_sub
#
# Figure 3b's "I-Norm (sub)" condition: the same local objective, but with the
# divisive inhibitory pathway removed, so only subtractive inhibition is
# available to match the LayerNorm statistics.
set -euo pipefail

# sbatch runs a spooled copy; fall back to the submit directory for siblings.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || true
if [[ ! -f "${_SCRIPT_DIR}/_common.sh" ]]; then
  _SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
source "${_SCRIPT_DIR}/_common.sh"

brightness_factors=(0 0.25 0.5 0.75)
bf=${brightness_factors[${SLURM_ARRAY_TASK_ID}]}

echo "grid=${SLURM_ARRAY_TASK_ID} brightness=$bf shunting=0"

export BRIGHTNESS_FACTOR=$bf
export NORMTYPE_DETACH=1
export LN_FEEDBACK="full"
export SHUNTING=0
export LAMBDA_HOMEOS=0.01

submit_inner_array "$SCRIPT_DIR/run_inorm_network.sh"
