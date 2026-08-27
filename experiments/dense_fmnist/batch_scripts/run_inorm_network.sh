#!/usr/bin/env bash
#SBATCH --array=0-29                 # 30 random hyperparameter configurations
#SBATCH --partition=long
#SBATCH --exclude=cn-c008
#SBATCH --gres=gpu:1
#SBATCH --constraint="turing|ampere|lovelace|hopper"  # skip volta (V100, CC 7.0): unsupported by our torch build
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/inorm_%A_%a.out
#SBATCH --error=sbatch_err/inorm_%A_%a.err
#SBATCH --job-name=inorm_homeo
set -euo pipefail

# sbatch runs a spooled copy; fall back to the submit directory for siblings.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || true
if [[ ! -f "${_SCRIPT_DIR}/_common.sh" ]]; then
  _SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
source "${_SCRIPT_DIR}/_common.sh"

read -r lr lr_wei lr_wix width < <(read_random_config "${SLURM_ARRAY_TASK_ID:-0}")

cd "$REPO_ROOT"
uv run python -m experiments.dense_fmnist.train_inorm_network \
  --train.dataset="${DATASET:-fashionmnist}" \
  --train.epochs="$EPOCHS" \
  --train.batch_size=32 \
  --data.data_dir="$DATA_DIR" \
  --data.brightness_factor="$BRIGHTNESS_FACTOR" \
  --model.normtype=0 \
  --model.normtype_detach="$NORMTYPE_DETACH" \
  --model.shunting="$SHUNTING" \
  --model.ln_feedback="$LN_FEEDBACK" \
  --model.excitation_training=1 \
  --model.hidden_layer_width="$width" \
  --model.homeostasis=1 \
  --model.track_alignment="${TRACK_ALIGNMENT:-1}" \
  --opt.lr="$lr" \
  --opt.inhib_lrs.wei="$lr_wei" \
  --opt.inhib_lrs.wix="$lr_wix" \
  --opt.momentum=0 \
  --opt.inhib_momentum=0 \
  --opt.use_sep_inhib_lrs=1 \
  --opt.lambda_homeo="$LAMBDA_HOMEOS" \
  --opt.lambda_homeo_var="$LAMBDA_HOMEOS" \
  --exp.use_wandb="${USE_WANDB:-1}" \
  --exp.wandb_project="$WANDB_PROJECT" \
  --exp.wandb_entity="$WANDB_ENTITY"
