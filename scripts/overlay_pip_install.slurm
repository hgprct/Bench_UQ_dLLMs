#!/bin/bash
#SBATCH --job-name=overlay_pip
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#
# Install pip packages into the writable overlay.
#
# Usage:
#   sbatch scripts/overlay_pip_install.slurm diffusers
#   sbatch scripts/overlay_pip_install.slurm "diffusers>=0.30" accelerate

set -euo pipefail

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }

PROJECT_DIR="${PROJECT_PATH:-$_dir}"
IMAGE="${CONTAINER_IMAGE:-$PROJECT_DIR/my_project.sif}"
OVERLAY="$(dirname "$IMAGE")/overlay.img"

if [ $# -eq 0 ]; then
  echo "ERROR: pass one or more pip packages as arguments" >&2
  exit 1
fi

if [ ! -f "$OVERLAY" ]; then
  echo "ERROR: overlay not found at $OVERLAY" >&2
  exit 1
fi

echo "=== Installing into $OVERLAY ==="
echo "Packages: $*"

apptainer exec \
  --overlay "${OVERLAY}" \
  --bind /lustre:/lustre \
  --pwd "$PROJECT_DIR" \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    pip install --no-cache-dir --break-system-packages "$@"
    echo "--- Installed versions ---"
    pip show "$@" 2>/dev/null | grep -E "^(Name|Version):" || true
  ' bash "$@"

echo "=== Done ==="
