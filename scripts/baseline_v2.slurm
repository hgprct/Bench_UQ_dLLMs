#!/bin/bash
#SBATCH --job-name=uq_baseline
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


set -euo pipefail

# Usage:
#   sbatch scripts/baseline_v2.slurm <run_dir>
#
# Examples:
#   sbatch scripts/baseline_v2.slurm outputs/LLaDA_triviaqa_l128_s64_lc
#
# Environment overrides:
#   NLI_MODEL=microsoft/deberta-v2-xlarge-mnli
#   NLI_BATCH_SIZE=32
#   TRAJECTORY_INDEX=0
#   SEED=42
#   FULL_SCOPE=1   (set to any non-empty value to enable full-trajectory NLI features)
#
# Always computed: token + iid-sample features, full-ad, random-scope (K=20, 5 reps)

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch scripts/baseline_v2.slurm <run_dir>"
  exit 1
fi

RUN_DIR="$1"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

BASELINE_ARGS=(
  --run_dir "$RUN_DIR"
)

[ -n "${NLI_MODEL:-}" ] && BASELINE_ARGS+=(--nli_model "$NLI_MODEL")
[ -n "${NLI_BATCH_SIZE:-}" ] && BASELINE_ARGS+=(--nli_batch_size "$NLI_BATCH_SIZE")
[ -n "${SEED:-}" ] && BASELINE_ARGS+=(--seed "$SEED")
[ -n "${FULL_SCOPE:-}" ] && BASELINE_ARGS+=(--full_scope)

print_uq_header "Baseline features: $RUN_DIR"

run_in_uq_container gpu python -m src.cli.baseline "${BASELINE_ARGS[@]}"

echo
echo "Baseline features finished at: $(date)"
