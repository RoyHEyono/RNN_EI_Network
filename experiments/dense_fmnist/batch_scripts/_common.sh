# Shared helpers for the dense Fashion-MNIST sweep scripts.
# Sourced by the run_*.sh and batch_*.sh scripts; not executable on its own.

# sbatch may execute a spooled copy of the script, so fall back to the
# submission directory when BASH_SOURCE does not resolve to a real path.
_resolve_script_dir() {
  local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  local dir
  dir="$(cd "$(dirname "$src")" 2>/dev/null && pwd)" || dir=""
  if [[ -z "$dir" || ! -f "$dir/_common.sh" ]]; then
    dir="${SLURM_SUBMIT_DIR:-$PWD}"
  fi
  printf '%s' "$dir"
}

SCRIPT_DIR="$(_resolve_script_dir)"
REPO_ROOT="${REPO_ROOT:-$HOME/RNN_EI_Network}"
RANDOM_CONFIGS_FILE="${RANDOM_CONFIGS_FILE:-$SCRIPT_DIR/random_configs.json}"
WANDB_PROJECT="${WANDB_PROJECT:-Luminosity_LNHomeostasis}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
EPOCHS="${EPOCHS:-50}"

# Pull one random hyperparameter config out of the JSON list by array index.
read_random_config() {
  local idx=$1
  uv run --directory "$REPO_ROOT" python - "$RANDOM_CONFIGS_FILE" "$idx" <<'PY'
import json, sys
configs = json.load(open(sys.argv[1]))
c = configs[int(sys.argv[2]) % len(configs)]
print(c["lr"], c["lr_wei"], c["lr_wix"], c["hidden_layer_width"])
PY
}

# Submit an inner array job without inheriting this job's SLURM array identity.
# Plain ``sbatch --export=ALL`` re-exports SLURM_ARRAY_TASK_ID (the outer grid
# index), which can corrupt nested ``#SBATCH --array=...`` ranges.
submit_inner_array() {
  local script=$1
  env -u SLURM_ARRAY_TASK_ID \
      -u SLURM_ARRAY_JOB_ID \
      -u SLURM_ARRAY_TASK_COUNT \
      -u SLURM_ARRAY_TASK_MAX \
      -u SLURM_ARRAY_TASK_MIN \
      -u SLURM_ARRAY_TASK_STEP \
      -u SLURM_JOB_ID \
      -u SLURM_JOBID \
      sbatch --export=ALL "$script"
}
