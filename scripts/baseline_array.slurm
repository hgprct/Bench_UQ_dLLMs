#!/bin/bash
#SBATCH --job-name=uq_baseline_arr
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-33%15
#SBATCH --exclude=dalianvl04,dalianvl08


set -euo pipefail

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

BASELINE_ARGS=(
  --run_dir "$RUN_DIR"
)

[ -n "${NLI_MODEL:-}" ] && BASELINE_ARGS+=(--nli_model "$NLI_MODEL")
[ -n "${NLI_BATCH_SIZE:-}" ] && BASELINE_ARGS+=(--nli_batch_size "$NLI_BATCH_SIZE")
[ -n "${SEED:-}" ] && BASELINE_ARGS+=(--seed "$SEED")
[ -n "${FULL_SCOPE:-}" ] && BASELINE_ARGS+=(--full_scope)

print_uq_header "Baseline features [${SLURM_ARRAY_TASK_ID}/${#RUN_DIRS[@]}]: $RUN_DIR"

run_in_uq_container gpu python -m src.cli.baseline "${BASELINE_ARGS[@]}"

echo
echo "Baseline features finished at: $(date)"
