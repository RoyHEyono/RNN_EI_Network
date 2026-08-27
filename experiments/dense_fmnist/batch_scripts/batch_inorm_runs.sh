#!/usr/bin/env bash
#SBATCH --array=0-31      # 4 luminosities x 2 detach x 4 feedback rules
#SBATCH --partition=long
#SBATCH --exclude=cn-c008
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"  # skip volta (V100, CC 7.0): unsupported by our torch build
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_inorm_%A_%a.out
#SBATCH --error=sbatch_err/grid_inorm_%A_%a.err
#SBATCH --job-name=grid_inorm
#
# Figures 3-6. normtype_detach=1 is plain I-Norm (Fig. 4); =0 adds GradNorm
# (Fig. 5), and ln_feedback selects which component of the LayerNorm gradient
# GradNorm imposes (Fig. 6).
set -euo pipefail

# sbatch runs a spooled copy; fall back to the submit directory for siblings.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || true
if [[ ! -f "${_SCRIPT_DIR}/_common.sh" ]]; then
  _SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
source "${_SCRIPT_DIR}/_common.sh"

brightness_factors=(0 0.25 0.5 0.75)
normtype_detach=(0 1)
ln_feedbacks=("scale" "center" "decorrelate" "full")

i=${SLURM_ARRAY_TASK_ID}
bf=${brightness_factors[$(( i % 4 ))]}
detach=${normtype_detach[$(( (i / 4) % 2 ))]}
fb=${ln_feedbacks[$(( (i / 8) % 4 ))]}

echo "grid=$i brightness=$bf normtype_detach=$detach ln_feedback=$fb"

export DATASET="${DATASET:-fashionmnist}"
export BRIGHTNESS_FACTOR=$bf
export NORMTYPE_DETACH=$detach
export LN_FEEDBACK=$fb
export SHUNTING=1
export LAMBDA_HOMEOS=0.01

submit_inner_array "$SCRIPT_DIR/run_inorm_network.sh"
