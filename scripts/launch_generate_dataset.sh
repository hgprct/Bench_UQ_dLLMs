#!/bin/bash
# Launch generate_v2.slurm for all configs matching a given dataset.
# Usage: ./scripts/launch_generate_dataset.sh <dataset> [--exclude-ceot]
#
# Examples:
#   ./scripts/launch_generate_dataset.sh gsm8k
#   ./scripts/launch_generate_dataset.sh triviaqa --exclude-ceot

set -euo pipefail

usage() {
    echo "Usage: $0 <dataset> [--exclude-ceot]" >&2
    echo "  <dataset>        Dataset name (e.g. gsm8k, triviaqa, wmt14_fr_en)" >&2
    echo "  --exclude-ceot   Skip configs with '_ceot' suffix (confidence_eos_eot_inf=true)" >&2
    exit 1
}

[[ $# -lt 1 ]] && usage

DATASET="$1"
EXCLUDE_CEOT=false

shift
for arg in "$@"; do
    case "$arg" in
        --exclude-ceot) EXCLUDE_CEOT=true ;;
        *) echo "Unknown argument: $arg" >&2; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs"

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -name "*_${DATASET}_*.json" | sort)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No config files found for dataset '${DATASET}' in ${CONFIG_DIR}" >&2
    exit 1
fi

submitted=0
skipped=0

for cfg in "${CONFIGS[@]}"; do
    basename="$(basename "$cfg" .json)"
    if $EXCLUDE_CEOT && [[ "$basename" == *_ceot ]]; then
        echo "Skipping (ceot): $basename"
        (( skipped++ )) || true
        continue
    fi
    echo "Submitting: $basename"
    sbatch "$SCRIPT_DIR/generate_v2.slurm" --config "$cfg"
    (( submitted++ )) || true
done

echo ""
echo "Submitted: ${submitted}  Skipped: ${skipped}"
