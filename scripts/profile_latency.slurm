#!/bin/bash
#SBATCH --job-name=uq_latency
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Usage:
#   sbatch scripts/profile_latency.slurm <config_path> [output_dir]
#
# Examples:
#   sbatch scripts/profile_latency.slurm configs/LLaDA_triviaqa_l32_s32_lc.json
#   sbatch scripts/profile_latency.slurm configs/LLaDA_triviaqa_l32_s32_lc.json outputs/latency/LLaDA_triviaqa
#
# Environment overrides:
#   NUM_PROMPTS=50          Number of timed prompts
#   WARMUP_PROMPTS=2        Number of warmup prompts (untimed)
#   SELECTION_DIR=...       Directory with trained_weights.npz
#   BUDGET_K=8              Top-K for QP weight selection
#   FULL_TRAJECTORY=1       Enable full-trajectory features (O(T^2) NLI; slow)
#   MAX_NLI_BATCH=128       Override autodetected NLI batch size
#   NLI_MODEL=microsoft/deberta-v2-xlarge-mnli
#   SEED=42

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch scripts/profile_latency.slurm <config_path> [output_dir]"
  exit 1
fi

CONFIG_PATH="$1"
CONFIG_STEM=$(basename "$CONFIG_PATH" .json)
OUTPUT_DIR="${2:-outputs/latency/$CONFIG_STEM}"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

LATENCY_ARGS=(
  --config "$CONFIG_PATH"
  --output_dir "$OUTPUT_DIR"
)

[ -n "${NUM_PROMPTS:-}" ] && LATENCY_ARGS+=(--num_prompts "$NUM_PROMPTS")
[ -n "${WARMUP_PROMPTS:-}" ] && LATENCY_ARGS+=(--warmup_prompts "$WARMUP_PROMPTS")
[ -n "${SELECTION_DIR:-}" ] && LATENCY_ARGS+=(--selection_dir "$SELECTION_DIR")
[ -n "${BUDGET_K:-}" ] && LATENCY_ARGS+=(--budget_k "$BUDGET_K")
[ -n "${FULL_TRAJECTORY:-}" ] && LATENCY_ARGS+=(--full_trajectory)
[ -n "${MAX_NLI_BATCH:-}" ] && LATENCY_ARGS+=(--max_nli_batch "$MAX_NLI_BATCH")
[ -n "${NLI_MODEL:-}" ] && LATENCY_ARGS+=(--nli_model "$NLI_MODEL")
[ -n "${SEED:-}" ] && LATENCY_ARGS+=(--seed "$SEED")

print_uq_header "Latency profiling: $CONFIG_STEM"
echo "Config:     $CONFIG_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Prompts:    ${NUM_PROMPTS:-50} (warmup: ${WARMUP_PROMPTS:-2})"
echo "Full traj:  ${FULL_TRAJECTORY:-disabled}"

run_in_uq_container gpu python -m src.cli.profile_latency "${LATENCY_ARGS[@]}"

echo
echo "Latency profiling finished at: $(date)"
