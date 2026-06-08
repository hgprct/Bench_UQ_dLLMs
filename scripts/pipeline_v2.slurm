#!/bin/bash
#SBATCH --job-name=uq_pipeline
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Usage:
#   sbatch scripts/pipeline_v2.slurm <model> <dataset> <num_response_samples> <length> <steps> <remasking>
#
# Models:   LLaDA, LLaDA1.5, Dream
# Datasets: triviaqa, gsm8k, wmt14_fr_en, wmt14_de_en, xsum, samsum
# Remasking: lc, rd
#
# Examples:
#   sbatch scripts/pipeline_v2.slurm LLaDA triviaqa 20 128 64 lc
#   sbatch scripts/pipeline_v2.slurm Dream gsm8k 20 128 64 rd
#
# Environment overrides:
#   NUM_QUESTIONS=1000 TEMPERATURE=1.0 BATCH_SIZE=8 SEED=42 FEWSHOT_K=5
#   LABEL_METHOD=exact_match JUDGE_MODEL=meta-llama/Llama-3.3-70B-Instruct JUDGE_TP=1
#   NLI_MODEL=microsoft/deberta-v2-xlarge-mnli
#   BOOTSTRAP_SAMPLES=1000 SKIP_MMD=true GENERATE_GREEDY=true|false

if [ "$#" -ne 6 ]; then
  echo "Usage: sbatch scripts/pipeline_v2.slurm <model> <dataset> <num_response_samples> <length> <steps> <lc|rd>"
  exit 1
fi

MODEL="$1"
DATASET="$2"
NUM_RESPONSE_SAMPLES="$3"
LENGTH="$4"
STEPS="$5"
REMASKING="$6"

# Validate canonical names
case "$MODEL" in
  LLaDA|LLaDA1.5|Dream) ;;
  *) echo "ERROR: model must be LLaDA, LLaDA1.5, or Dream (got '$MODEL')"; exit 1 ;;
esac

case "$DATASET" in
  triviaqa|gsm8k|wmt14_fr_en|wmt14_de_en|xsum|samsum) ;;
  *) echo "ERROR: dataset must be one of: triviaqa, gsm8k, wmt14_fr_en, wmt14_de_en, xsum, samsum (got '$DATASET')"; exit 1 ;;
esac

case "$REMASKING" in
  lc|rd) ;;
  *) echo "ERROR: remasking must be lc or rd (got '$REMASKING')"; exit 1 ;;
esac

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

# Build pipeline args
PIPELINE_ARGS=(
  --model "$MODEL"
  --dataset "$DATASET"
  --num_response_samples "$NUM_RESPONSE_SAMPLES"
  --length "$LENGTH"
  --steps "$STEPS"
  --remasking "$REMASKING"
)

[ -n "${NUM_QUESTIONS:-}" ] && PIPELINE_ARGS+=(--num_questions "$NUM_QUESTIONS")
[ -n "${TEMPERATURE:-}" ] && PIPELINE_ARGS+=(--temperature "$TEMPERATURE")
[ -n "${BATCH_SIZE:-}" ] && PIPELINE_ARGS+=(--batch_size "$BATCH_SIZE")
[ -n "${SEED:-}" ] && PIPELINE_ARGS+=(--seed "$SEED")
[ -n "${LABEL_METHOD:-}" ] && PIPELINE_ARGS+=(--label_method "$LABEL_METHOD")
[ -n "${JUDGE_MODEL:-}" ] && PIPELINE_ARGS+=(--judge_model "$JUDGE_MODEL")
[ -n "${JUDGE_TP:-}" ] && PIPELINE_ARGS+=(--judge_tp "$JUDGE_TP")
[ -n "${NLI_MODEL:-}" ] && PIPELINE_ARGS+=(--nli_model "$NLI_MODEL")
[ -n "${NLI_BATCH_SIZE:-}" ] && PIPELINE_ARGS+=(--nli_batch_size "$NLI_BATCH_SIZE")
[ -n "${BOOTSTRAP_SAMPLES:-}" ] && PIPELINE_ARGS+=(--bootstrap_samples "$BOOTSTRAP_SAMPLES")
[ -n "${FEWSHOT_K:-}" ] && PIPELINE_ARGS+=(--fewshot_k "$FEWSHOT_K")
[ "${SKIP_MMD:-false}" = "true" ] && PIPELINE_ARGS+=(--skip_mmd)
[ "${GENERATE_GREEDY:-}" = "true" ] && PIPELINE_ARGS+=(--generate_greedy)
[ "${GENERATE_GREEDY:-}" = "false" ] && PIPELINE_ARGS+=(--no-generate_greedy)

print_uq_header "Pipeline: $MODEL $DATASET l${LENGTH} s${STEPS} ${REMASKING}"

run_in_uq_container gpu python -m src.cli.pipeline "${PIPELINE_ARGS[@]}"

echo
echo "Pipeline finished at: $(date)"
