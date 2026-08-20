#!/usr/bin/env bash
#SBATCH --array=0-3%2       # 4 luminosities
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/grid_inorm_sub_%A_%a.out
#SBATCH --error=sbatch_err/grid_inorm_sub_%A_%a.err
#SBATCH --job-name=grid_inorm_sub
#
# Figure 3b's "I-Norm (sub)" condition: the same local objective, but with the
# divisive inhibitory pathway removed, so only subtractive inhibition is
# available to match the LayerNorm statistics.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

brightness_factors=(0 0.25 0.5 0.75)
bf=${brightness_factors[${SLURM_ARRAY_TASK_ID}]}

echo "grid=${SLURM_ARRAY_TASK_ID} brightness=$bf shunting=0"

export BRIGHTNESS_FACTOR=$bf
export NORMTYPE_DETACH=1
export LN_FEEDBACK="full"
export SHUNTING=0
export LAMBDA_HOMEOS=0.01

sbatch --export=ALL "$SCRIPT_DIR/run_inorm_network.sh"
