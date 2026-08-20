#!/usr/bin/env bash
#SBATCH --array=0-29                # 30 random hyperparameter configurations
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/inorm_%A_%a.out
#SBATCH --error=sbatch_err/inorm_%A_%a.err
#SBATCH --job-name=inorm_homeo
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
source "${REPO_ROOT}/.venv/bin/activate"
PYTHON_BIN=python

read -r lr lr_wei lr_wix width < <(read_random_config "${SLURM_ARRAY_TASK_ID:-0}")

cd "$REPO_ROOT"
python -m experiments.dense_fmnist.train_inorm_network \
  --train.dataset='fashionmnist' \
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
