#!/bin/bash
# Launch LLaDA1.5 generation across all 7 datasets, both remaskings.
#
# Three phases:
#   1. Generate config JSONs (per-dataset max_gen_length/steps/block_size from DATASET_CONFIGS)
#   2. Build prompts.jsonl for each config (CPU container, may download datasets)
#   3. Submit SLURM array job for GPU generation
#
# Default generation: 1 greedy per prompt (no stochastic), 1000 questions.
# Batch size is auto-calibrated at runtime to fit GPU memory.
#
# Usage:
#   bash scripts/launch_llada15_gen.sh [options]
#
# Options:
#   --concurrency N     Max simultaneous SLURM tasks (default: 6)
#   --output-dir DIR    Parent output directory (default: outputs)
#   --dry-run           Print commands without submitting
#   --prompts-only      Only build prompts, don't submit generation jobs
#   --num-questions N   Override number of questions per dataset (default: 1000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs"

CONCURRENCY=14
OUTPUT_DIR="outputs"
DRY_RUN=false
PROMPTS_ONLY=false
NUM_QUESTIONS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency)    CONCURRENCY="$2"; shift ;;
    --output-dir)     OUTPUT_DIR="$2";  shift ;;
    --num-questions)  NUM_QUESTIONS="$2"; shift ;;
    --dry-run)        DRY_RUN=true ;;
    --prompts-only)   PROMPTS_ONLY=true ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Load .env and container env if available (for running prompt prep in container)
[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }
source "$SCRIPT_DIR/slurm_env.sh"
init_uq_slurm_env

# ─── Phase 1: Generate config files ────────────────────────────────────────
echo "=== Phase 1: Generating config files ==="

# Remove stale LLaDA1.5 configs so the Phase 2 glob only finds freshly generated ones
rm -f "$CONFIG_DIR"/LLaDA1.5_*.json

python3 -m src.cli.gen_configs \
  --model LLaDA1.5 \
  --dataset triviaqa wmt14_fr_en xsum samsum hotpotqa musique gsm8k \
  --remasking lc rd \
  --num_questions "$NUM_QUESTIONS" \
  --num_response_samples 0 \
  --temperature 0.0 \
  --overwrite

# ─── Phase 2: Build prompts ────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Building prompts ==="

CONFIG_LIST="$CONFIG_DIR/llada15_gen.txt"
find "$CONFIG_DIR" -maxdepth 1 -name "LLaDA1.5_*.json" | sort > "$CONFIG_LIST"
N=$(wc -l < "$CONFIG_LIST")

echo "Found $N config files"
cat "$CONFIG_LIST"
echo ""

if [[ $N -eq 0 ]]; then
  echo "ERROR: no LLaDA1.5 configs found in $CONFIG_DIR" >&2
  exit 1
fi

# Build prompts inside the container on a compute node (needs HuggingFace datasets)
mapfile -t CONFIGS < "$CONFIG_LIST"

PROMPT_CMD="python scripts/prepare_all_prompts.py --output-dir $OUTPUT_DIR ${CONFIGS[*]}"

if $DRY_RUN; then
  echo "[dry-run] sbatch --wait build_prompts.slurm -- $PROMPT_CMD"
else
  echo "Submitting prompt-build job (sbatch --wait) ..."
  sbatch --wait \
    --job-name=build_prompts \
    --output="/lustre/work/pdl16831/%u/stage/logs/%x_%j.txt" \
    --error="/lustre/work/pdl16831/%u/stage/logs/%x_%j_err.txt" \
    --time=00:30:00 \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --wrap="
      set -euo pipefail
      [ -f '$PROJECT_DIR/.env' ] && { set -a; source '$PROJECT_DIR/.env'; set +a; }
      source '$SCRIPT_DIR/slurm_env.sh'
      init_uq_slurm_env
      run_in_uq_container cpu $PROMPT_CMD
    "
  echo "Prompt-build job finished."
fi

echo ""
echo "=== Prompt summary ==="
for config in "${CONFIGS[@]}"; do
  run_id="$(basename "$config" .json)"
  prompts_file="$OUTPUT_DIR/$run_id/prompts.jsonl"
  if [[ -f "$prompts_file" ]]; then
    count=$(wc -l < "$prompts_file")
    echo "  $run_id: $count prompts"
  else
    echo "  $run_id: MISSING prompts.jsonl"
  fi
done

if $PROMPTS_ONLY; then
  echo ""
  echo "=== Done (prompts-only mode) ==="
  exit 0
fi

# ─── Phase 3: Submit SLURM array ───────────────────────────────────────────
echo ""
echo "=== Phase 3: Submitting SLURM array ==="

ARRAY_UPPER=$((N - 1))
ARRAY_SPEC="0-${ARRAY_UPPER}%${CONCURRENCY}"

echo "Array spec: ${ARRAY_SPEC}  (${N} tasks, max ${CONCURRENCY} concurrent)"
echo "Output dir: $OUTPUT_DIR"
echo ""

if $DRY_RUN; then
  echo "[dry-run] CONFIGS_FILE=$CONFIG_LIST OUTPUT_DIR=$OUTPUT_DIR \\"
  echo "  sbatch --array=$ARRAY_SPEC $SCRIPT_DIR/generate_array.slurm"
else
  JOB=$(
    CONFIGS_FILE="$CONFIG_LIST" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    sbatch --array="$ARRAY_SPEC" \
      "$SCRIPT_DIR/generate_array.slurm" \
      | tee /dev/stderr | grep -oP '\d+'
  )
  echo ""
  echo "Submitted job ${JOB}  (${N} tasks, ${CONCURRENCY} concurrent)"
  echo "Monitor:  squeue -j ${JOB}"
  echo "Logs:     /lustre/work/pdl16831/\$USER/stage/logs/uq_generate_arr_${JOB}_*.out"
  echo "Outputs:  $OUTPUT_DIR"
fi
