#!/bin/bash
#SBATCH --job-name=dg_mathvision
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#
# Run DiffusionGemma on a few MathVision examples, using the isolated venv built
# by scripts/diffusiongemma_setup.slurm (transformers 5.x layered over the
# container's torch; shared overlay left untouched).
#
# Usage:  sbatch scripts/diffusiongemma_mathvision_demo.slurm --num 3 --image-tokens 1120

set -euo pipefail
_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }

PROJECT_DIR="${PROJECT_PATH:-$_dir}"
STAGE_DIR="${STAGE_DIR:-${STAGE_DIR_PATH:?STAGE_DIR_PATH not set}}"
IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE not set}"
OVERLAY="$(dirname "$IMAGE")/overlay.img"
VENV="${STAGE_DIR}/envs/diffusiongemma"

if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: venv missing at $VENV -- run scripts/diffusiongemma_setup.slurm first" >&2
  exit 1
fi

echo "============================================================"
echo "DiffusionGemma x MathVision"
echo "Host: $(hostname)  Arch: $(uname -m)  Job: ${SLURM_JOB_ID:-none}"
echo "Image: $IMAGE"
echo "Venv:  $VENV"
echo "Args:  $*"
echo "============================================================"

ov=(); [ -f "$OVERLAY" ] && ov=(--overlay "$OVERLAY:ro")

apptainer exec --nv "${ov[@]}" --bind /lustre:/lustre --pwd "$PROJECT_DIR" "$IMAGE" \
  bash -lc '
    set -euo pipefail
    unset TRANSFORMERS_CACHE
    export PYTHONPATH="'"$PROJECT_DIR"'${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    source "'"$VENV"'/bin/activate"
    python -c "import transformers; print(\"transformers\", transformers.__version__)"
    python scripts/diffusiongemma_mathvision_demo.py "$@"
  ' bash "$@"

echo "Finished at: $(date)"
