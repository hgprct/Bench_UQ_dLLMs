#!/bin/bash
#SBATCH --job-name=uq_latency_arr
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-29%10

set -euo pipefail

# Usage:
#   sbatch scripts/profile_latency_array.slurm <runs_dir>
#
# Profiles latency for every run subdirectory inside <runs_dir>.
# Each task uses the config.json from the run subdir and the
# trained_weights.npz from <run_dir>/quickeval/selection/ (if present).
#
# Output goes to <run_dir>/latency/.
#
# Examples:
#   sbatch scripts/profile_latency_array.slurm outputs_old
#   sbatch scripts/profile_latency_array.slurm outputs
#
# Environment overrides (same as profile_latency.slurm):
#   NUM_PROMPTS=50          Number of timed prompts
#   WARMUP_PROMPTS=2        Number of warmup prompts (untimed)
#   BUDGET_K=8              Top-K for QP weight selection
#   FULL_TRAJECTORY=1       Enable full-trajectory features (O(T^2) NLI; slow)
#   MAX_NLI_BATCH=128       Override autodetected NLI batch size
#   NLI_MODEL=microsoft/deberta-v2-xlarge-mnli
#   SEED=42
#   SELECTION_SUBDIR=quickeval/selection   Subdir holding trained_weights.npz

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch scripts/profile_latency_array.slurm <runs_dir>"
  exit 1
fi

RUNS_DIR="$1"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

# Make RUNS_DIR absolute if relative
[[ "$RUNS_DIR" != /* ]] && RUNS_DIR="${_dir}/${RUNS_DIR}"

mapfile -t RUN_DIRS < <(ls -1 "$RUNS_DIR" | sort)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#RUN_DIRS[@]}" ]; then
  echo "Task ID ${SLURM_ARRAY_TASK_ID} exceeds number of runs (${#RUN_DIRS[@]}), exiting."
  exit 0
fi

RUN_DIR="${RUNS_DIR}/${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"
CONFIG_PATH="${RUN_DIR}/config.json"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: config not found: $CONFIG_PATH"
  exit 1
fi

OUTPUT_DIR="${RUN_DIR}/latency"

SELECTION_SUBDIR="${SELECTION_SUBDIR:-quickeval/selection}"
SELECTION_DIR="${RUN_DIR}/${SELECTION_SUBDIR}"

LATENCY_ARGS=(
  --config "$CONFIG_PATH"
  --output_dir "$OUTPUT_DIR"
)

if [ -f "${SELECTION_DIR}/trained_weights.npz" ]; then
  LATENCY_ARGS+=(--selection_dir "$SELECTION_DIR")
else
  echo "WARNING: No trained_weights.npz at ${SELECTION_DIR}; selected-* features will be skipped"
fi

[ -n "${NUM_PROMPTS:-}" ] && LATENCY_ARGS+=(--num_prompts "$NUM_PROMPTS")
[ -n "${WARMUP_PROMPTS:-}" ] && LATENCY_ARGS+=(--warmup_prompts "$WARMUP_PROMPTS")
[ -n "${BUDGET_K:-}" ] && LATENCY_ARGS+=(--budget_k "$BUDGET_K")
[ -n "${FULL_TRAJECTORY:-}" ] && LATENCY_ARGS+=(--full_trajectory)
[ -n "${MAX_NLI_BATCH:-}" ] && LATENCY_ARGS+=(--max_nli_batch "$MAX_NLI_BATCH")
[ -n "${NLI_MODEL:-}" ] && LATENCY_ARGS+=(--nli_model "$NLI_MODEL")
[ -n "${SEED:-}" ] && LATENCY_ARGS+=(--seed "$SEED")

print_uq_header "Latency profiling [${SLURM_ARRAY_TASK_ID}/${#RUN_DIRS[@]}]: $RUN_DIR"
echo "Config:       $CONFIG_PATH"
echo "Output:       $OUTPUT_DIR"
echo "Selection:    ${SELECTION_DIR}"
echo "Prompts:      ${NUM_PROMPTS:-50} (warmup: ${WARMUP_PROMPTS:-2})"
echo "Full traj:    ${FULL_TRAJECTORY:-disabled}"

run_in_uq_container gpu python -m src.cli.profile_latency "${LATENCY_ARGS[@]}"

echo
echo "Latency profiling finished at: $(date)"
