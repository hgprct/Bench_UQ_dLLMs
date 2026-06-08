#!/bin/bash
# Run label_all_v2, then launch both baseline_array jobs in parallel once it finishes.
#
# Usage:
#   bash scripts/label_then_baseline.sh [outputs_folder]

set -euo pipefail

FOLDER="${1:-outputs/}"

# 1. Submit label_all — capture its job ID with --parsable
LABEL_JID=$(sbatch --parsable scripts/label_all_v2.slurm "$FOLDER")
echo "label_all       → job $LABEL_JID"

# 2. Submit both baseline_array jobs, dependent on label_all succeeding.
#    Both share the same dependency so they start in parallel.
B1_JID=$(RANDOM_SCOPE=1 sbatch --parsable \
  --dependency=afterok:"$LABEL_JID" \
  scripts/baseline_array.slurm)
echo "baseline_array  → job $B1_JID  (depends on $LABEL_JID)"

B2_JID=$(RANDOM_SCOPE=1 RANDOM_SCOPE_K=100 sbatch --parsable \
  --dependency=afterok:"$LABEL_JID" \
  scripts/baseline_array.slurm)
echo "baseline_array  → job $B2_JID  (depends on $LABEL_JID, RANDOM_SCOPE_K=100)"

echo ""
echo "Chain: $LABEL_JID → [$B1_JID, $B2_JID]"
echo "Monitor: squeue -u \"\$USER\""
