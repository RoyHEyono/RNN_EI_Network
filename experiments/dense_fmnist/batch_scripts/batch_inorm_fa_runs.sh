#!/usr/bin/env bash
#SBATCH --array=0-7      # 4 luminosities x 2 detach
#SBATCH --partition=long
#SBATCH --exclude=cn-c008
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"  # skip volta (V100, CC 7.0): unsupported by our torch build
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_fa_%A_%a.out
#SBATCH --error=sbatch_err/grid_fa_%A_%a.err
#SBATCH --job-name=grid_fa
#
# Figure 7: gradient centering performed by a lateral inhibitory pool with
# fixed, random, positive synapses (ln_feedback=fa_center).
set -euo pipefail

# sbatch runs a spooled copy; fall back to the submit directory for siblings.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || true
if [[ ! -f "${_SCRIPT_DIR}/_common.sh" ]]; then
  _SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
source "${_SCRIPT_DIR}/_common.sh"

brightness_factors=(0 0.25 0.5 0.75)
normtype_detach=(0 1)

i=${SLURM_ARRAY_TASK_ID}
bf=${brightness_factors[$(( i % 4 ))]}
detach=${normtype_detach[$(( (i / 4) % 2 ))]}

echo "grid=$i brightness=$bf normtype_detach=$detach ln_feedback=fa_center"

export DATASET="${DATASET:-fashionmnist}"
export BRIGHTNESS_FACTOR=$bf
export NORMTYPE_DETACH=$detach
export LN_FEEDBACK="fa_center"
export SHUNTING=1
export LAMBDA_HOMEOS=0.01

submit_inner_array "$SCRIPT_DIR/run_inorm_network.sh"
