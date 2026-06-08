#!/bin/bash
set -euo pipefail

# Launch the full generate → label → baseline → quickeval pipeline with Slurm dependencies.
#
# Usage:
#   bash scripts/launch_pipeline.sh <configs_file> <n_train>
#
# Arguments:
#   configs_file   Path to a text file with one config JSON path per line
#   n_train        Number of training samples for quickeval (e.g. 10)
#
# Environment overrides:
#   OUTPUT_DIR       Parent output directory (default: outputs_new)
#   MAX_CONCURRENT   Max simultaneous array tasks (default: 6)
#   LAMBDA_VALUES    Space-separated lambda values for quickeval (default: "0.0 0.01 0.1 0.2 0.5 1.0 5.0 10.0")
#   N_VAL / N_TEST   Split sizes for quickeval
#   SEED             Random seed
#   NLI_MODEL / NLI_BATCH_SIZE
#   JUDGE_MODEL / JUDGE_TP / JUDGE_MAX_SEQS / JUDGE_GPU_MEM  (label_all)
#   FORCE_LABEL      Set to 1 to re-label even if already labeled
#
# Example:
#   bash scripts/launch_pipeline.sh configs/my_list.txt 10
#   OUTPUT_DIR=outputs_exp1 MAX_CONCURRENT=4 bash scripts/launch_pipeline.sh configs/my_list.txt 20

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/launch_pipeline.sh <configs_file> <n_train>"
  exit 1
fi

CONFIGS_FILE="$1"
N_TRAIN="$2"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_new}"
MAX_CONCURRENT="${MAX_CONCURRENT:-6}"

if [ ! -f "$CONFIGS_FILE" ]; then
  echo "ERROR: configs file not found: $CONFIGS_FILE"
  exit 1
fi

N_CONFIGS=$(grep -c -v '^\s*$' "$CONFIGS_FILE")
if [ "$N_CONFIGS" -eq 0 ]; then
  echo "ERROR: no configs found in $CONFIGS_FILE"
  exit 1
fi

ARRAY_RANGE="0-$((N_CONFIGS - 1))%${MAX_CONCURRENT}"

_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"

# Resolve OUTPUT_DIR to absolute path
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$_dir/$OUTPUT_DIR"
fi

echo "============================================================"
echo "Pipeline launcher"
echo "Configs:     $CONFIGS_FILE ($N_CONFIGS configs)"
echo "Output dir:  $OUTPUT_DIR"
echo "n_train:     $N_TRAIN"
echo "Array:       $ARRAY_RANGE"
echo "============================================================"

# 1. Generate array
GEN_JID=$(sbatch --parsable \
  --array="$ARRAY_RANGE" \
  --export=ALL,CONFIGS_FILE="$(realpath "$CONFIGS_FILE")",OUTPUT_DIR="$OUTPUT_DIR" \
  "$_dir/scripts/generate_array.slurm")
echo "generate_array submitted: $GEN_JID  (array $ARRAY_RANGE)"

# 2. Label all (single job, runs after the full generate array)
LABEL_JID=$(sbatch --parsable \
  --dependency=afterok:"$GEN_JID" \
  "$_dir/scripts/label_all_v2.slurm" "$OUTPUT_DIR")
echo "label_all submitted:      $LABEL_JID  (after $GEN_JID)"

# 3. Baseline array
BASELINE_JID=$(sbatch --parsable \
  --dependency=afterok:"$LABEL_JID" \
  --array="$ARRAY_RANGE" \
  --export=ALL,RUNS_DIR="$OUTPUT_DIR" \
  "$_dir/scripts/baseline_array.slurm")
echo "baseline_array submitted: $BASELINE_JID  (after $LABEL_JID)"

# 4. Quickeval array
QUICKEVAL_JID=$(sbatch --parsable \
  --dependency=afterok:"$BASELINE_JID" \
  --array="$ARRAY_RANGE" \
  --export=ALL,RUNS_DIR="$OUTPUT_DIR" \
  "$_dir/scripts/quickeval_array.slurm" "$N_TRAIN")
echo "quickeval_array submitted: $QUICKEVAL_JID  (after $BASELINE_JID)"

echo
echo "Dependency chain: $GEN_JID -> $LABEL_JID -> $BASELINE_JID -> $QUICKEVAL_JID"
echo "Monitor with: squeue -u \"\$USER\""
