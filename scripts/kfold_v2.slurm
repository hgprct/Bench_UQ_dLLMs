#!/bin/bash
#SBATCH --job-name=uq_kfold
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Usage:
#   sbatch scripts/kfold_v2.slurm <run_dir> <n_train> <baseline_features_path>
#
# Examples:
#   sbatch scripts/kfold_v2.slurm outputs/LLaDA_triviaqa_l128_s64_lc 10 outputs/LLaDA_triviaqa_l128_s64_lc/baseline_features.jsonl
#
# Environment overrides:
#   N_FOLDS=5 SEED=42 BUDGET_K=8
#   LAMBDA_VALUES="0.001 0.01 0.1 1.0 10.0"
#   EMBEDDING_MODEL=all-MiniLM-L6-v2
#   NLI_MODEL=microsoft/deberta-v2-xlarge-mnli
#   NLI_BATCH_SIZE=8
#   BOOTSTRAP_SAMPLES=1000
#   SELECTION_MODE=all_sampling SELECTION_METRIC=auroc
#   ANCHOR_FEATURE=se-marginal   # single feature for val lambda selection (overrides SELECTION_MODE)
#   SKIP_NLI=true FULL_TRAJ=true

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch scripts/kfold_v2.slurm <run_dir> <n_train> <baseline_features_path>"
  exit 1
fi

RUN_DIR="$1"
N_TRAIN="$2"
BASELINE_FEATURES_PATH="$3"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

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

LAMBDA_VALUES="${LAMBDA_VALUES:-0.1}"
KFOLD_ARGS+=(--lambda_values $LAMBDA_VALUES)

print_uq_header "K-fold CV: $RUN_DIR (n_train=$N_TRAIN, baseline=$BASELINE_FEATURES_PATH)"

run_in_uq_container gpu python -m src.cli.kfold "${KFOLD_ARGS[@]}"

echo
echo "K-fold finished at: $(date)"
