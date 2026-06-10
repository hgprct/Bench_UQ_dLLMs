#!/bin/bash

set -euo pipefail

# ── Resolve PROJECT_DIR ──────────────────────────────────────────────────
# Prefer PROJECT_PATH from .env, then fall back to filesystem detection.
if [[ -n "${PROJECT_PATH:-}" ]]; then
  export PROJECT_DIR="$PROJECT_PATH"
elif [[ -z "${PROJECT_DIR:-}" ]]; then
  _senv_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export PROJECT_DIR="$(dirname "$_senv_dir")"
fi

# ── Resolve STAGE_DIR ────────────────────────────────────────────────────
# .env sets STAGE_DIR_PATH; bashrc sets STAGE_DIR. Accept either.
export STAGE_DIR="${STAGE_DIR:-${STAGE_DIR_PATH:-}}"

init_uq_slurm_env() {
  if [[ -z "$STAGE_DIR" ]]; then
    echo "ERROR: STAGE_DIR (or STAGE_DIR_PATH in .env) is not set." >&2
    exit 1
  fi

  # Use CONTAINER_IMAGE from .env when available; otherwise derive from
  # UQ_IMAGE variant (dlm / nemotron).
  if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    export IMAGE="$CONTAINER_IMAGE"
    export OVERLAY="$(dirname "$IMAGE")/overlay.img"
  else
    local image_variant="${UQ_IMAGE:-dlm}"
    case "$image_variant" in
      dlm)
        export IMAGE="${STAGE_DIR}/code/Bench_UQ_dLLMs/my_project.sif"
        export OVERLAY="${STAGE_DIR}/code/Bench_UQ_dLLMs/overlay.img"
        ;;
      nemotron)
        export IMAGE="${STAGE_DIR}/code/Bench_UQ_dLLMs/my_project_nemotron.sif"
        export OVERLAY="${STAGE_DIR}/code/Bench_UQ_dLLMs/overlay_nemotron.img"
        ;;
      *)
        echo "ERROR: unknown UQ_IMAGE='$image_variant'. Use 'dlm' or 'nemotron'." >&2
        exit 1
        ;;
    esac
  fi

  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  unset TRANSFORMERS_CACHE

  mkdir -p "${STAGE_DIR}/logs"

  cd "$PROJECT_DIR"
}

print_uq_header() {
  local title="$1"
  echo "============================================================"
  echo "$title"
  echo "============================================================"
  echo "Date:        $(date)"
  echo "Job ID:      ${SLURM_JOB_ID:-none}"
  echo "Host:        $(hostname)"
  echo "Arch:        $(uname -m)"
  echo "Project dir: $PROJECT_DIR"
  echo "Image:       $IMAGE"
  echo "============================================================"
}

run_in_uq_container() {
  local gpu_mode="$1"
  shift

  local apptainer_gpu_args=()
  if [ "$gpu_mode" = "gpu" ]; then
    # --nv est crucial pour lier le GPU Grace Blackwell au conteneur
    apptainer_gpu_args=(--nv)
  fi

  # Lancement pur et dur. Plus aucune installation à la volée.
  local overlay_args=()
  if [ -f "${OVERLAY:-}" ]; then
    overlay_args=(--overlay "${OVERLAY}:ro")
  fi

  apptainer exec \
    "${apptainer_gpu_args[@]}" \
    "${overlay_args[@]}" \
    --bind /lustre:/lustre \
    --pwd "$PROJECT_DIR" \
    "$IMAGE" \
    bash -lc '
      set -euo pipefail
      unset TRANSFORMERS_CACHE
      
      # On ajoute le dossier racine au PYTHONPATH pour que "src" soit détecté
      export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
      export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export VLLM_USE_V1=0
      
      cd "$PROJECT_DIR"
      
      # Exécution de la commande python passée en argument
      "$@"
    ' bash "$@"
}

run_stage() {
  local stage_name="$1"
  shift

  echo
  echo "============================================================"
  echo "STARTING: $stage_name"
  echo "Time:     $(date)"
  echo "Command:  $*"
  echo "============================================================"

  "$@"

  echo "============================================================"
  echo "FINISHED: $stage_name"
  echo "Time:     $(date)"
  echo "============================================================"
}