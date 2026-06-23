#!/bin/bash
# Submit ONE generation job per model, smoke-testing each model on a couple of
# prompts of every dataset available to it. Raw answers only -- no judge, no eval.
#
# Each job runs scripts/smoke_all_models.slurm for a single MODEL, which loops
# over that model's compatible datasets (text models -> text datasets; vision
# models -> image datasets) and writes raw traces to
#   $OUTBASE/<MODEL>/<dataset>/answers.jsonl
#
# Usage:
#   bash scripts/launch_smoke_all.sh [options]
#
# Options:
#   --models "A B C"    Space-separated model list (default: the 5 benched models)
#   --num-questions N   Prompts per dataset (default: 2)
#   --outbase DIR       Output root (default: outputs/smoke_all)
#   --time HH:MM:SS     Walltime per job (default: 02:00:00)
#   --dry-run           Print sbatch commands without submitting
#
# Note: LLaDA (base) is omitted by default -- it shares the `llada` backend with
# LLaDA1.5. Add it with: --models "LLaDA LLaDA1.5 Dream Nemotron MMaDA NemotronVLM"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODELS=(LLaDA1.5 Dream Nemotron MMaDA NemotronVLM)
NUM_QUESTIONS=2
OUTBASE="outputs/smoke_all"
TIME="02:00:00"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)         read -r -a MODELS <<< "$2"; shift ;;
    --num-questions)  NUM_QUESTIONS="$2"; shift ;;
    --outbase)        OUTBASE="$2"; shift ;;
    --time)           TIME="$2"; shift ;;
    --dry-run)        DRY_RUN=true ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

echo "Models:        ${MODELS[*]}"
echo "Prompts/ds:    $NUM_QUESTIONS"
echo "Output root:   $OUTBASE"
echo "Walltime/job:  $TIME"
echo

for m in "${MODELS[@]}"; do
  cmd=(sbatch
    --job-name="smoke_${m}"
    --time="$TIME"
    --export="ALL,MODEL=${m},NUM_QUESTIONS=${NUM_QUESTIONS},OUTBASE=${OUTBASE}"
    "$SCRIPT_DIR/smoke_all_models.slurm")
  if $DRY_RUN; then
    echo "[dry-run] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
done

if ! $DRY_RUN; then
  echo
  echo "Submitted ${#MODELS[@]} jobs (one per model)."
  echo "Monitor:  squeue -u \$USER"
  echo "Logs:     /lustre/work/pdl16831/\$USER/stage/logs/smoke_<model>_<jobid>.out"
  echo "Outputs:  $OUTBASE/<model>/<dataset>/answers.jsonl"
fi
