#!/bin/bash
# Launch a SLURM job array: LLaDA1.5, all 8 datasets, both remaskings, l64_s64.
#
# Each task runs one config through src.cli.generate (greedy + 20 sampled, full traces).
# Configs are generated on-the-fly if they do not exist yet, then listed in a
# stable file that the array tasks read at runtime.
#
# Usage:
#   bash scripts/launch_llada15_l64.sh [options]
#
# Options:
#   --concurrency N   Max simultaneous jobs (default: 4)
#   --output-dir DIR  Parent output directory (default: outputs/LLaDA1.5_l64_s64)
#   --dry-run         Print the sbatch command without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs"

CONCURRENCY=15
OUTPUT_DIR="outputs/LLaDA1.5_l64_s64"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --concurrency)  CONCURRENCY="$2"; shift ;;
    --output-dir)   OUTPUT_DIR="$2";  shift ;;
    --dry-run)      DRY_RUN=true ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Step 1: generate config files (pure Python, no GPU needed)
# ---------------------------------------------------------------------------
echo "=== Generating configs ==="

# All datasets except gsm8k use fewshot_k=0
python3 -m src.cli.gen_configs \
  --model LLaDA1.5 \
  --dataset triviaqa wmt14_fr_en wmt14_de_en xsum samsum hotpotqa musique \
  --remasking lc rd \
  --length 64 --steps 64 \
  --overwrite

# gsm8k uses fewshot_k=4 (matches existing convention and FEWSHOT_K=4 in the adapter)
python3 -m src.cli.gen_configs \
  --model LLaDA1.5 \
  --dataset gsm8k \
  --remasking lc rd \
  --length 64 --steps 64 \
  --fewshot_k 4 \
  --overwrite

# ---------------------------------------------------------------------------
# Step 2: build the config list (one path per line, sorted)
# ---------------------------------------------------------------------------
CONFIG_LIST="$CONFIG_DIR/llada15_l64_s64.txt"
find "$CONFIG_DIR" -maxdepth 1 -name "LLaDA1.5_*_l64_s64*.json" | sort > "$CONFIG_LIST"
N=$(wc -l < "$CONFIG_LIST")

echo ""
echo "=== Config list ($N configs) → $CONFIG_LIST ==="
cat "$CONFIG_LIST"
echo ""

if [[ $N -eq 0 ]]; then
  echo "ERROR: no LLaDA1.5 l64_s64 configs found in $CONFIG_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: submit the array
# ---------------------------------------------------------------------------
ARRAY_UPPER=$((N - 1))
ARRAY_SPEC="0-${ARRAY_UPPER}%${CONCURRENCY}"

echo "=== Submitting array=${ARRAY_SPEC} (output: $OUTPUT_DIR) ==="

if $DRY_RUN; then
  echo "[dry-run] CONFIGS_FILE=$CONFIG_LIST OUTPUT_DIR=$OUTPUT_DIR \\"
  echo "  sbatch --array=$ARRAY_SPEC $SCRIPT_DIR/generate_array.slurm --num_questions 10 --temperature 0"
else
  JOB=$(
    CONFIGS_FILE="$CONFIG_LIST" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    sbatch --array="$ARRAY_SPEC" \
      "$SCRIPT_DIR/generate_array.slurm" \
      --num_questions 10 \
      --temperature 0 \
      | tee /dev/stderr | grep -oP '\d+'
  )
  echo ""
  echo "Submitted job ${JOB}  (${N} tasks, ${CONCURRENCY} concurrent)"
  echo "Monitor:  squeue -j ${JOB}"
  echo "Logs:     /lustre/work/pdl16831/\$USER/stage/logs/uq_generate_arr_${JOB}_*.out"
  echo "Outputs:  $OUTPUT_DIR"
fi
