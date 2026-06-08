#!/bin/bash
#SBATCH --job-name=uq_quickeval
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Usage:
#   sbatch scripts/quickeval_v2.slurm <run_dir> <n_train> <baseline_features_path>
#
# Examples:
#   sbatch scripts/quickeval_v2.slurm outputs/LLaDA_triviaqa_l128_s64_lc 10 outputs/LLaDA_triviaqa_l128_s64_lc/baseline_features.jsonl
#
# Environment overrides:
#   N_VAL=30 N_TEST=40            Split sizes (auto if unset; ignored for single lambda)
#   SEED=42 BUDGET_K=8
#   LAMBDA_VALUES="0.1"           Single lambda merges val into test
#   EMBEDDING_MODEL=all-MiniLM-L6-v2
#   NLI_MODEL=microsoft/deberta-v2-xlarge-mnli
#   NLI_BATCH_SIZE=8
#   BOOTSTRAP_SAMPLES=1000
#   SELECTION_MODE=all_sampling SELECTION_METRIC=prr
#   ANCHOR_FEATURE=eigval
#   SKIP_NLI=true FULL_TRAJ=true

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch scripts/quickeval_v2.slurm <run_dir> <n_train> <baseline_features_path>"
  exit 1
fi

RUN_DIR="$1"
N_TRAIN="$2"
BASELINE_FEATURES_PATH="$3"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

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

LAMBDA_VALUES="${LAMBDA_VALUES:-0.0}"
QUICKEVAL_ARGS+=(--lambda_values $LAMBDA_VALUES)

print_uq_header "Quick eval: $RUN_DIR (n_train=$N_TRAIN, baseline=$BASELINE_FEATURES_PATH)"

run_in_uq_container gpu python -m src.cli.quickeval "${QUICKEVAL_ARGS[@]}"

echo
echo "Quick eval finished at: $(date)"
