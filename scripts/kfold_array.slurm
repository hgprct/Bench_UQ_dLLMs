#!/bin/bash
#SBATCH --job-name=uq_kfold_arr
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-33%33
#SBATCH --exclude=dalianvl04,dalianvl08


set -euo pipefail

# Usage:
#   sbatch scripts/kfold_array.slurm <n_train>
#
# Runs kfold on all outputs/ directories.
# Each task picks its baseline_features.jsonl from the run dir.
#
# Environment overrides: same as kfold_v2.slurm (including ANCHOR_FEATURE)

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch scripts/kfold_array.slurm <n_train>"
  exit 1
fi

N_TRAIN="$1"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

RUNS_DIR="${_dir}/outputs"
mapfile -t RUN_DIRS < <(ls -1 "$RUNS_DIR" | sort)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#RUN_DIRS[@]}" ]; then
  echo "Task ID ${SLURM_ARRAY_TASK_ID} exceeds number of runs (${#RUN_DIRS[@]}), exiting."
  exit 0
fi

RUN_DIR="${RUNS_DIR}/${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"
BASELINE_FEATURES_PATH="${RUN_DIR}/baseline_features.jsonl"

if [ ! -f "$BASELINE_FEATURES_PATH" ]; then
  echo "ERROR: baseline features not found: $BASELINE_FEATURES_PATH"
  echo "Run baseline_array.slurm first."
  exit 1
fi

KFOLD_ARGS=(
  --run_dir "$RUN_DIR"
  --n_train "$N_TRAIN"
  --baseline_features_path "$BASELINE_FEATURES_PATH"
)

[ -n "${N_FOLDS:-}" ] && KFOLD_ARGS+=(--n_folds "$N_FOLDS")
[ -n "${SEED:-}" ] && KFOLD_ARGS+=(--seed "$SEED")
[ -n "${BUDGET_K:-}" ] && KFOLD_ARGS+=(--budget_k "$BUDGET_K")
[ -n "${EMBEDDING_MODEL:-}" ] && KFOLD_ARGS+=(--embedding_model "$EMBEDDING_MODEL")
[ -n "${NLI_MODEL:-}" ] && KFOLD_ARGS+=(--nli_model "$NLI_MODEL")
[ -n "${NLI_BATCH_SIZE:-}" ] && KFOLD_ARGS+=(--nli_batch_size "$NLI_BATCH_SIZE")
[ -n "${BOOTSTRAP_SAMPLES:-}" ] && KFOLD_ARGS+=(--bootstrap_samples "$BOOTSTRAP_SAMPLES")
[ -n "${SELECTION_MODE:-}" ] && KFOLD_ARGS+=(--selection_mode "$SELECTION_MODE")
[ -n "${SELECTION_METRIC:-}" ] && KFOLD_ARGS+=(--selection_metric "$SELECTION_METRIC")
[ -n "${ANCHOR_FEATURE:-}" ] && KFOLD_ARGS+=(--anchor_feature "$ANCHOR_FEATURE")
[ "${SKIP_NLI:-false}" = "true" ] && KFOLD_ARGS+=(--skip_nli)
[ "${FULL_TRAJ:-false}" = "true" ] && KFOLD_ARGS+=(--full_traj)

LAMBDA_VALUES="${LAMBDA_VALUES:-0.0 0.01 0.1 0.2 0.5 1.0 5.0 10.0 50.0}"
KFOLD_ARGS+=(--lambda_values $LAMBDA_VALUES)

print_uq_header "K-fold CV [${SLURM_ARRAY_TASK_ID}/${#RUN_DIRS[@]}]: $RUN_DIR (n_train=$N_TRAIN)"

run_in_uq_container gpu python -m src.cli.kfold "${KFOLD_ARGS[@]}"

echo
echo "K-fold finished at: $(date)"
