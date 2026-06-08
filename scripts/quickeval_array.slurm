#!/bin/bash
#SBATCH --job-name=uq_quickeval_arr
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-30%6
#SBATCH --exclude=dalianvl04

set -euo pipefail

# Usage:
#   sbatch scripts/quickeval_array.slurm <n_train>
#
# Runs quickeval on all outputs_old/ directories.
# Each task picks its baseline_features.jsonl from the run dir.
#
# Environment overrides: same as quickeval_v2.slurm

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch scripts/quickeval_array.slurm <n_train>"
  exit 1
fi

N_TRAIN="$1"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

RUNS_DIR="${RUNS_DIR:-${_dir}/outputs}"
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

QUICKEVAL_ARGS=(
  --run_dir "$RUN_DIR"
  --n_train "$N_TRAIN"
  --baseline_features_path "$BASELINE_FEATURES_PATH"
)

[ -n "${N_VAL:-}" ] && QUICKEVAL_ARGS+=(--n_val "$N_VAL")
[ -n "${N_TEST:-}" ] && QUICKEVAL_ARGS+=(--n_test "$N_TEST")
[ -n "${SEED:-}" ] && QUICKEVAL_ARGS+=(--seed "$SEED")
[ -n "${BUDGET_K:-}" ] && QUICKEVAL_ARGS+=(--budget_k "$BUDGET_K")
[ -n "${EMBEDDING_MODEL:-}" ] && QUICKEVAL_ARGS+=(--embedding_model "$EMBEDDING_MODEL")
[ -n "${NLI_MODEL:-}" ] && QUICKEVAL_ARGS+=(--nli_model "$NLI_MODEL")
[ -n "${NLI_BATCH_SIZE:-}" ] && QUICKEVAL_ARGS+=(--nli_batch_size "$NLI_BATCH_SIZE")
[ -n "${BOOTSTRAP_SAMPLES:-}" ] && QUICKEVAL_ARGS+=(--bootstrap_samples "$BOOTSTRAP_SAMPLES")
[ -n "${SELECTION_MODE:-}" ] && QUICKEVAL_ARGS+=(--selection_mode "$SELECTION_MODE")
[ -n "${SELECTION_METRIC:-}" ] && QUICKEVAL_ARGS+=(--selection_metric "$SELECTION_METRIC")
[ -n "${ANCHOR_FEATURE:-}" ] && QUICKEVAL_ARGS+=(--anchor_feature "$ANCHOR_FEATURE")
[ "${SKIP_NLI:-false}" = "true" ] && QUICKEVAL_ARGS+=(--skip_nli)
[ "${FULL_TRAJ:-false}" = "true" ] && QUICKEVAL_ARGS+=(--full_traj)

LAMBDA_VALUES="${LAMBDA_VALUES:-0.0 0.01 0.1 0.2 0.5 1.0 5.0 10.0}"
QUICKEVAL_ARGS+=(--lambda_values $LAMBDA_VALUES)

print_uq_header "Quick eval [${SLURM_ARRAY_TASK_ID}/${#RUN_DIRS[@]}]: $RUN_DIR (n_train=$N_TRAIN)"

run_in_uq_container gpu python -m src.cli.quickeval "${QUICKEVAL_ARGS[@]}"

echo
echo "Quick eval finished at: $(date)"
