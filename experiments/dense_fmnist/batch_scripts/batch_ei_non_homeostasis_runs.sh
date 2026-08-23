#!/usr/bin/env bash
#SBATCH --array=0-31%2      # 4 luminosities x 2 normtype x 2 detach x 2 architecture
#SBATCH --partition=long
#SBATCH --exclude=cn-c008
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"  # skip volta (V100, CC 7.0): unsupported by our torch build
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_ei_%A_%a.out
#SBATCH --error=sbatch_err/grid_ei_%A_%a.err
#SBATCH --job-name=grid_ei
#
# Figure 2: LN vs no-LN in E/I networks, and E/I vs E-only under LN.
# Each grid point fans out into 5 random hyperparameter configurations.
set -euo pipefail

# sbatch runs a spooled copy; fall back to the submit directory for siblings.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || true
if [[ ! -f "${_SCRIPT_DIR}/_common.sh" ]]; then
  _SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
source "${_SCRIPT_DIR}/_common.sh"

brightness_factors=(0 0.25 0.5 0.75)
layer_norms=(0 1)
normtype_detach=(0 1)
excitatory_only=(0 1)

i=${SLURM_ARRAY_TASK_ID}
bf=${brightness_factors[$(( i % 4 ))]}
ln=${layer_norms[$(( (i / 4) % 2 ))]}
detach=${normtype_detach[$(( (i / 8) % 2 ))]}
eonly=${excitatory_only[$(( (i / 16) % 2 ))]}

echo "grid=$i brightness=$bf layer_norm=$ln normtype_detach=$detach excitatory_only=$eonly"

export BRIGHTNESS_FACTOR=$bf
export LAYER_NORM=$ln
export NORMTYPE=0
export NORMTYPE_DETACH=$detach
export EXCITATORY_ONLY=$eonly
export LAMBDA_HOMEOS=1        # inert: no homeostatic loss in this arm

submit_inner_array "$SCRIPT_DIR/run_ei_network.sh"
