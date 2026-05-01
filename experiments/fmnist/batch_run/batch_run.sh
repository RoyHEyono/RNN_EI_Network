#!/usr/bin/env bash
#SBATCH --array=0-19  # 20 random configurations
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/random_config_%A_%a.out
#SBATCH --error=sbatch_err/random_config_%A_%a.err
#SBATCH --job-name=EERNN

# Load environment
source /home/mila/r/roy.eyono/RNN_EI_Network/.venv/bin/activate

# Fixed data augmentation (Fashion-MNIST brightness jitter; see experiments/cli.py)
BRIGHTNESS_FACTOR=0.75

# Load random parameters from file
random_configs_file='random_configs.json'
random_index=$SLURM_ARRAY_TASK_ID
random_params=$(python -c "import json; import sys; f=open('$random_configs_file'); configs=json.load(f); f.close(); print(json.dumps(configs[$random_index]))")
lr=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['lr'])")
lr_wei=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['lr_ei'])")
lr_wix=$(echo $random_params | python -c "import sys, json; config=json.load(sys.stdin); print(config['lr_ie'])")

# Run your training script with the specific parameters
uv run python /home/mila/r/roy.eyono/RNN_EI_Network/experiments/main.py \
  --lr=$lr \
  --lr-ie=$lr_wei \
  --lr-ei=$lr_wix \
  --brightness-factor=$BRIGHTNESS_FACTOR