#!/bin/bash
#SBATCH --job-name=uq_label_all
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=dalianvl02,dalianvl03,dalianvl04


source ~/.bashrc 2>/dev/null || true
set -euo pipefail

# Label every run in a folder in a single job, loading the LLM judge once.
#
# Usage:
#   sbatch scripts/label_all_v2.slurm [folder]
#
# Arguments:
#   folder   Directory containing run subdirectories (default: outputs/)
#
# Environment overrides:
#   JUDGE_MODEL    vLLM judge model (default: meta-llama/Llama-3.3-70B-Instruct)
#   JUDGE_TP       vLLM tensor parallel size
#   JUDGE_MAX_SEQS vLLM max concurrent sequences (default: vLLM auto)
#   JUDGE_GPU_MEM  vLLM GPU memory utilization 0-1 (default: vLLM auto)
#   EXCLUDE_RUNS   Space-separated run directory names to skip
#   SEED           Random seed (default: 42)
#   PRINT_EXAMPLES Set to 1 to print per-run accuracy after labeling
#   FORCE_LABEL    Set to 1 to re-label all records even if already labeled

FOLDER="${1:-outputs/}"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

if [[ "$FOLDER" != /* ]]; then
  FOLDER="$PROJECT_DIR/$FOLDER"
fi

if [ ! -d "$FOLDER" ]; then
  echo "ERROR: folder '${FOLDER}' does not exist"
  exit 1
fi

print_uq_header "Label All: ${FOLDER}"

LABEL_ARGS=(--folder "$FOLDER")
[ -n "${JUDGE_MODEL:-}" ] && LABEL_ARGS+=(--judge-model "$JUDGE_MODEL")
[ -n "${JUDGE_TP:-}" ]       && LABEL_ARGS+=(--judge-tp "$JUDGE_TP")
[ -n "${JUDGE_MAX_SEQS:-}" ] && LABEL_ARGS+=(--judge-max-num-seqs "$JUDGE_MAX_SEQS")
[ -n "${JUDGE_GPU_MEM:-}" ]  && LABEL_ARGS+=(--judge-gpu-mem "$JUDGE_GPU_MEM")
[ -n "${SEED:-}" ]           && LABEL_ARGS+=(--seed "$SEED")
[ "${PRINT_EXAMPLES:-0}" = "1" ] && LABEL_ARGS+=(--print-examples)
[ "${FORCE_LABEL:-0}" = "1" ]   && LABEL_ARGS+=(--force)

if [ -n "${EXCLUDE_RUNS:-}" ]; then
  # shellcheck disable=SC2206
  EXCLUDE_ARRAY=($EXCLUDE_RUNS)
  LABEL_ARGS+=(--exclude "${EXCLUDE_ARRAY[@]}")
fi

run_in_uq_container gpu python -m src.cli.label_all "${LABEL_ARGS[@]}"

echo
echo "All labeling finished at: $(date)"
