#!/bin/bash
set -euo pipefail

# Submit one label_v2.slurm job per run in outputs/.
#
# Usage (from login node):
#   bash scripts/relabel_all.sh
#
# Environment overrides (forwarded to each job):
#   LABEL_METHOD, JUDGE_MODEL, JUDGE_TP, SEED
#
# Note: llm_judge runs each load the judge model independently.
# For many llm_judge runs, consider submitting them manually as a
# single label_v2.slurm call with multiple --run-dir arguments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RUNS=()
for d in outputs/*/; do
    [ -f "${d}config.json" ] && [ -f "${d}examples.jsonl" ] && RUNS+=("$d")
done

if [ ${#RUNS[@]} -eq 0 ]; then
    echo "No runs found in outputs/. Nothing to do."
    exit 0
fi

echo "Found ${#RUNS[@]} runs to relabel:"
printf "  %s\n" "${RUNS[@]}"
echo

EXPORT_VARS=()
[ -n "${LABEL_METHOD:-}" ] && EXPORT_VARS+=(LABEL_METHOD="$LABEL_METHOD")
[ -n "${JUDGE_MODEL:-}" ]  && EXPORT_VARS+=(JUDGE_MODEL="$JUDGE_MODEL")
[ -n "${JUDGE_TP:-}" ]     && EXPORT_VARS+=(JUDGE_TP="$JUDGE_TP")
[ -n "${SEED:-}" ]         && EXPORT_VARS+=(SEED="$SEED")

EXPORT_FLAG=""
if [ ${#EXPORT_VARS[@]} -gt 0 ]; then
    EXPORT_FLAG="--export=ALL,$(IFS=,; echo "${EXPORT_VARS[*]}")"
fi

for d in "${RUNS[@]}"; do
    run_name=$(basename "$d")
    # shellcheck disable=SC2086
    sbatch --job-name="uq_label_${run_name}" \
           ${EXPORT_FLAG} \
           scripts/label_v2.slurm "$d"
done

echo
echo "Submitted ${#RUNS[@]} labeling jobs."
