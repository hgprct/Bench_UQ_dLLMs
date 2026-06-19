#!/bin/bash
#SBATCH --job-name=dg_setup
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#
# One-time setup for DiffusionGemma:
#   1. Create an ISOLATED venv (--system-site-packages, so it inherits the
#      container's torch) and install transformers>=5.8 + accelerate into it.
#      This does NOT touch the shared overlay used by the MMaDA/LLaDA jobs.
#   2. Pre-download the 51 GB google/diffusiongemma-26B-A4B-it snapshot into HF_HOME.
#
# Usage:  sbatch scripts/diffusiongemma_setup.slurm

set -euo pipefail
_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }

PROJECT_DIR="${PROJECT_PATH:-$_dir}"
STAGE_DIR="${STAGE_DIR:-${STAGE_DIR_PATH:?STAGE_DIR_PATH not set}}"
IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE not set}"
OVERLAY="$(dirname "$IMAGE")/overlay.img"
VENV="${STAGE_DIR}/envs/diffusiongemma"
TF_VERSION="${TF_VERSION:-5.12.1}"
MODEL_ID="google/diffusiongemma-26B-A4B-it"

echo "=== DiffusionGemma setup ==="
echo "Host:    $(hostname)  Arch: $(uname -m)"
echo "Image:   $IMAGE"
echo "Venv:    $VENV   (transformers==$TF_VERSION)"
echo "HF_HOME: ${HF_HOME:-unset}"

ov=(); [ -f "$OVERLAY" ] && ov=(--overlay "$OVERLAY:ro")

apptainer exec "${ov[@]}" --bind /lustre:/lustre --pwd "$PROJECT_DIR" "$IMAGE" \
  bash -lc '
    set -euo pipefail
    VENV="'"$VENV"'"; TF_VERSION="'"$TF_VERSION"'"; MODEL_ID="'"$MODEL_ID"'"

    if [ ! -x "$VENV/bin/python" ]; then
      echo "--- creating venv (inherits container torch) ---"
      python -m venv --system-site-packages "$VENV"
    fi
    source "$VENV/bin/activate"
    # The NGC base image pins many packages via a PIP_CONSTRAINT file (e.g.
    # regex==2024.11.6) that conflicts with transformers 5.x. We are installing
    # into an isolated venv, so drop those constraints for this install only.
    unset PIP_CONSTRAINT PIP_CONFIG_FILE 2>/dev/null || true
    python -m pip install --no-cache-dir --upgrade pip

    echo "--- installing transformers==$TF_VERSION + accelerate + pillow (no torch pull) ---"
    pip install --no-cache-dir "transformers==$TF_VERSION" "accelerate>=1.1.0" pillow

    echo "--- verifying versions / diffusion_gemma classes ---"
    python - <<PY
import torch, transformers
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor  # noqa
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print("diffusion_gemma in CONFIG_MAPPING:", "diffusion_gemma" in CONFIG_MAPPING_NAMES)
print("classes import OK")
PY

    echo "--- pre-downloading $MODEL_ID snapshot (~51 GB) into HF cache ---"
    python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_ID")
print("snapshot at:", p)
PY
  '

echo "=== Setup done at $(date) ==="
