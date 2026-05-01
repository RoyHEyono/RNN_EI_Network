#!/usr/bin/env bash
#SBATCH --array=0-39
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/neurogym_%A_%a.out
#SBATCH --error=sbatch_err/neurogym_%A_%a.err
#SBATCH --job-name=ng-ei-rnn

REPO_ROOT=/home/mila/r/roy.eyono/RNN_EI_Network

source "${REPO_ROOT}/.venv/bin/activate"

TASK="${TASK:-ContextDecisionMaking-v0}"
_wandb_slug=$(echo "$TASK" | tr '[:upper:]' '[:lower:]' | sed -e 's/[^a-z0-9]\+/\-/g' -e 's/^-\|-$//g')
WANDB_PROJECT="${WANDB_PROJECT:-ng-${_wandb_slug}}"

# Load random parameters (submit from this directory, like fmnist/batch_run)
random_configs_file='random_configs.json'
random_index=$SLURM_ARRAY_TASK_ID
random_params=$(python -c "import json; import sys; f=open('$random_configs_file'); configs=json.load(f); f.close(); print(json.dumps(configs[$random_index]))")
lr=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['lr'])")
seed=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['seed'])")
arch=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config.get('arch', 'ei'))")

if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  export WANDB_RUN_NAME="ng_${SLURM_ARRAY_JOB_ID}_${arch}_${random_index}"
else
  export WANDB_RUN_NAME="ng_manual_${arch}_${random_index}"
fi

cd "$REPO_ROOT" || exit 1

uv run --extra neurogym python -m experiments.neurogym.main \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --task "$TASK" \
  --arch "$arch" \
  --optimizer sgd \
  --lr=$lr \
  --seed=$seed
