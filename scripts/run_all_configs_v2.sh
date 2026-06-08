#!/bin/bash
# Run the full pipeline for all configs in the configs/ directory.
#
# Usage:
#   bash scripts/run_all_configs_v2.sh [NUM_RESPONSE_SAMPLES]
#
# Submits one Slurm job per config file matching the naming convention.

set -euo pipefail

NUM_SAMPLES="${1:-20}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for config in configs/*_l*_s*_*.json; do
  [ -f "$config" ] || continue
  basename="$(basename "$config" .json)"

  # Parse: Model_dataset_lL_sS_rm (dataset may contain underscores, e.g. wmt14_fr_en)
  MODEL="$(echo "$basename" | cut -d_ -f1)"
  rest="${basename#${MODEL}_}"
  DATASET="$(echo "$rest" | grep -oP '^.*(?=_l\d+_)')"
  LENGTH="$(echo "$basename" | grep -oP 'l\K\d+')"
  STEPS="$(echo "$basename" | grep -oP 's\K\d+')"
  REMASKING="$(echo "$basename" | grep -oP '(lc|rd)$')"

  if [ -z "$MODEL" ] || [ -z "$DATASET" ] || [ -z "$LENGTH" ] || [ -z "$STEPS" ] || [ -z "$REMASKING" ]; then
    echo "SKIP: cannot parse $config"
    continue
  fi

  echo "Submitting: $MODEL $DATASET l${LENGTH} s${STEPS} ${REMASKING}"
  sbatch "$SCRIPT_DIR/pipeline_v2.slurm" "$MODEL" "$DATASET" "$NUM_SAMPLES" "$LENGTH" "$STEPS" "$REMASKING"
done
