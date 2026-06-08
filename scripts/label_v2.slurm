#!/bin/bash
#SBATCH --job-name=uq_label
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=dalianvl02,dalianvl03,dalianvl04

source ~/.bashrc 2>/dev/null || true
set -euo pipefail

# Label a single run directory.
#
# Usage:
#   sbatch scripts/label_v2.slurm <run_dir>
#   sbatch scripts/label_v2.slurm outputs/LLaDA_triviaqa_l128_s64_lc/

if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch scripts/label_v2.slurm <run_dir>"
  exit 1
fi

RUN_DIR="$1"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$PROJECT_DIR/$RUN_DIR"
fi

if [ ! -f "${RUN_DIR}/config.json" ] || [ ! -f "${RUN_DIR}/examples.jsonl" ]; then
  echo "ERROR: ${RUN_DIR} must contain config.json and examples.jsonl"
  exit 1
fi

print_uq_header "Label: ${RUN_DIR}"

LABEL_ARGS=(--run-dir "$RUN_DIR" --print-examples)
[ -n "${LABEL_METHOD:-}" ]  && LABEL_ARGS+=(--method "$LABEL_METHOD")
[ -n "${JUDGE_MODEL:-}" ]   && LABEL_ARGS+=(--judge-model "$JUDGE_MODEL")
[ -n "${JUDGE_TP:-}" ]         && LABEL_ARGS+=(--judge-tp "$JUDGE_TP")
[ -n "${JUDGE_MAX_SEQS:-}" ]   && LABEL_ARGS+=(--judge-max-num-seqs "$JUDGE_MAX_SEQS")
[ -n "${JUDGE_GPU_MEM:-}" ]    && LABEL_ARGS+=(--judge-gpu-mem "$JUDGE_GPU_MEM")
[ -n "${SEED:-}" ]             && LABEL_ARGS+=(--seed "$SEED")
[ "${FORCE_LABEL:-0}" = "1" ] && LABEL_ARGS+=(--force)

# Ceci va maintenant utiliser notre environnement gelé sans aucun temps mort
run_in_uq_container gpu python -m src.cli.label "${LABEL_ARGS[@]}"

echo
echo "Labeling finished at: $(date)"