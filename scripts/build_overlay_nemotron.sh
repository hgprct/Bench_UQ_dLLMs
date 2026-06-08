#!/bin/bash
# Build a writable overlay that upgrades transformers 4.x → 5.x
# for Nemotron support. Runs on top of the existing my_project.sif.
#
# Usage (on a compute node via srun, never on login):
#   srun --gres=gpu:0 --cpus-per-task=4 --time=01:00:00 \
#        bash scripts/build_overlay_nemotron.sh [size_mb]
#
# Or submit as a Slurm job:
#   sbatch scripts/build_overlay_nemotron.slurm

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

IMAGE="$PROJECT_DIR/my_project.sif"
OVERLAY_PATH="$PROJECT_DIR/overlay_nemotron.img"
SIZE_MB="${1:-1024}"

if [ ! -f "$IMAGE" ]; then
  echo "ERROR: base image not found at $IMAGE" >&2
  exit 1
fi

if [ -f "$OVERLAY_PATH" ]; then
  echo "WARNING: $OVERLAY_PATH already exists, backing up to ${OVERLAY_PATH}.bak"
  mv "$OVERLAY_PATH" "${OVERLAY_PATH}.bak"
fi

echo "=== Creating ${SIZE_MB}MB ext3 overlay ==="
dd if=/dev/zero of="$OVERLAY_PATH" bs=1M count="$SIZE_MB"
mkfs.ext3 -F "$OVERLAY_PATH"

echo "=== Installing transformers 5.x into overlay ==="
apptainer exec \
  --overlay "${OVERLAY_PATH}" \
  --bind /lustre:/lustre \
  --pwd "$PROJECT_DIR" \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    pip install --no-cache-dir --break-system-packages \
      "transformers>=5.0" \
      "accelerate>=1.0" \
      "tokenizers>=0.22"
    echo "--- Installed versions ---"
    python -c "
import transformers, accelerate, tokenizers
print(f\"transformers {transformers.__version__}\")
print(f\"accelerate   {accelerate.__version__}\")
print(f\"tokenizers   {tokenizers.__version__}\")
"
  '

echo "=== Done: $OVERLAY_PATH ==="
echo "Use with: UQ_IMAGE=nemotron sbatch scripts/generate_v2.slurm ..."
