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

echo "=== Installing transformers 5.x into overlay prefix /opt/nemo5 ==="
# We layer ONLY the transformers-5.x stack on top of the NGC base, into a
# SEPARATE prefix /opt/nemo5 via `pip --target`. Why a separate prefix and not a
# plain `pip install`: a normal install UNINSTALLS the base transformers/
# tokenizers (4.55.4 / 0.21.4). Under a read-only overlay, apptainer stacks the
# ext3 image as a lower layer where overlayfs whiteouts are NOT honoured, so the
# uninstalled base dist-info stays visible and transformers-5's own import-time
# version check sees tokenizers 0.21.4 and aborts. `pip --target` only ADDS
# files (pure addition is shown correctly under :ro); runtime then puts
# /opt/nemo5 first on PYTHONPATH (see slurm_env.sh) so 5.x + its metadata win.
#
# The NGC base already ships the Grace-Blackwell torch (2.7.0a0+...nv25.4),
# flash-attn aarch64, numpy 1.26, pillow, opencv and requests, all reused as-is.
# We pin torch+numpy to the in-image versions so the resolver can't drag a
# different torch / numpy>=2 into the prefix (which would break the NGC torch
# ABI); it fails loudly here rather than corrupting the env at runtime.
#
# --fakeroot is required to write INTO the overlay: unprivileged overlay
# upper-dir creation is denied on the compute nodes (same reason the base .sif
# is built with `apptainer build --fakeroot`). Runtime use mounts it `:ro` and
# needs no fakeroot.
apptainer exec \
  --fakeroot \
  --overlay "${OVERLAY_PATH}" \
  --bind /lustre:/lustre \
  --pwd "$PROJECT_DIR" \
  "$IMAGE" \
  bash -c '
    set -euo pipefail

    BASE_TORCH="$(python -c "import torch; print(torch.__version__)")"
    BASE_NUMPY="$(python -c "import numpy; print(numpy.__version__)")"
    echo "Base (NGC) torch=${BASE_TORCH}  numpy=${BASE_NUMPY}"
    # Pin ONLY numpy (a findable PyPI version). The NGC torch is a LOCAL build
    # whose exact version string is on no index, so pinning it makes the pip
    # --target resolver fail the moment a torch-requiring package (accelerate) is
    # in the set. transformers/tokenizers do NOT hard-require torch, so we put
    # just those in the prefix; accelerate (base 1.13.0) resolves from the base
    # image at runtime (it is always on sys.path, just after /opt/nemo5).
    printf "numpy==%s\n" "$BASE_NUMPY" > /tmp/nemotron_constraints.txt

    # Pin transformers to 4.57.1 -- the version that satisfies BOTH Nemotron
    # repos (the /opt/nemo5 dir name is historical; it holds 4.57.1):
    #   * VLM-8B (config 4.57.1) imports transformers.masking_utils.
    #     sdpa_mask_older_torch, a 4.x symbol removed in 5.x -> 5.0.0 ImportError.
    #   * text-14B (config 5.0.0) imports utils.generic.check_model_inputs (added
    #     ~4.52, present in 4.57.1) and uses llama-4 rope scaling
    #     (llama_4_scaling_beta) that 5.0.0 mishandles -> garbage logits -> the
    #     threshold-driven diffusion never converges and hangs. 4.57.1 handles it.
    # Let tokenizers resolve to whatever 4.57.1 requires (do not force a range).
    pip install --no-cache-dir --break-system-packages \
      --target /opt/nemo5 \
      -c /tmp/nemotron_constraints.txt \
      "transformers==4.57.1"

    echo "--- Verifying overlay (/opt/nemo5 first on sys.path) ---"
    # No f-strings / no backslash-escapes here: the heredoc is unquoted and lives
    # inside a single-quoted `bash -c`, so plain double quotes pass through fine
    # while escaped ones (\") would reach python verbatim and break the f-string.
    PYTHONPATH="/opt/nemo5" python - <<PY
import importlib.metadata as md
import torch, transformers, accelerate, tokenizers
tv = transformers.__version__; tvm = md.version("transformers")
tkv = tokenizers.__version__; tkm = md.version("tokenizers")
print("transformers", tv, "metadata", tvm)
print("tokenizers  ", tkv, "metadata", tkm)
print("accelerate  ", accelerate.__version__)
print("torch       ", torch.__version__)
print("transformers from:", transformers.__file__)
# Hard guards: transformers must resolve to 4.57.x (module AND metadata) from
# /opt/nemo5, BOTH symbols the two Nemotron repos import must resolve, NGC torch kept.
assert tv.startswith("4.57"), tv
assert tvm.startswith("4.57"), tvm
assert "/opt/nemo5/" in transformers.__file__, transformers.__file__
from transformers.utils.generic import check_model_inputs  # text-14B needs this
from transformers.masking_utils import sdpa_mask_older_torch  # VLM-8B needs this
print("check_model_inputs + sdpa_mask_older_torch import: OK")
assert "nv" in torch.__version__, torch.__version__
# VLM image-preprocessing deps are provided by the base image; confirm import.
import PIL, cv2, requests
print("pillow", PIL.__version__, "opencv", cv2.__version__, "requests", requests.__version__)
print("OK: overlay is consistent")
PY
  '

echo "=== Done: $OVERLAY_PATH ==="
echo "Run Nemotron with:"
echo "  CONTAINER_OVERLAY=$OVERLAY_PATH sbatch scripts/generate_v2.slurm --model Nemotron ..."
