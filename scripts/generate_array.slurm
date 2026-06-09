#!/bin/bash
#SBATCH --job-name=uq_generate_arr
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-30%6
#SBATCH --exclude=dalianvl04

set -euo pipefail

# Usage (via a launcher script, not directly):
#   CONFIGS_FILE=configs/my_list.txt OUTPUT_DIR=outputs_new sbatch --array=0-N%6 scripts/generate_array.slurm [extra args]
#
# CONFIGS_FILE: path to a text file with one config JSON path per line (blank lines ignored)
# OUTPUT_DIR:   parent output directory (default: outputs_new); run_id subdirs are created inside
# extra args:   forwarded verbatim to src.cli.generate (e.g. --num_questions 10)

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

if [ -z "${CONFIGS_FILE:-}" ]; then
  echo "ERROR: CONFIGS_FILE env var is required"
  exit 1
fi

mapfile -t CONFIGS < <(grep -v '^\s*$' "$CONFIGS_FILE")

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#CONFIGS[@]}" ]; then
  echo "Task ID ${SLURM_ARRAY_TASK_ID} exceeds number of configs (${#CONFIGS[@]}), exiting."
  exit 0
fi

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
RUN_ID="$(basename "$CONFIG" .json)"

OUTPUT_DIR="${OUTPUT_DIR:-outputs_new}"
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"
fi

print_uq_header "Generate [${SLURM_ARRAY_TASK_ID}/${#CONFIGS[@]}]: $CONFIG -> $OUTPUT_DIR/$RUN_ID"

run_in_uq_container gpu python -m src.cli.generate \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR/$RUN_ID" \
  "$@"

echo
echo "Generation finished at: $(date)"
