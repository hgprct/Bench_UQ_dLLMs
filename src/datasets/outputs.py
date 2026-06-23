"""Output/disk layer for the dataset pipeline.

Layout (shared by every model)::

    outputs/
      <dataset>/
        images/
          <image_id>.png         # saved once at data-prep time, reused thereafter
        <model_name>.jsonl        # one clean per-prompt record file per model

The per-model JSONL carries exactly the canonical :class:`QASample` contract --
no model-specific framing, no top-k logprobs (those live in the per-run UQ traces
written by ``src.generate.traces``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample
from src.utils.io import write_jsonl

DEFAULT_OUTPUTS_ROOT = "outputs"

# The canonical, serialisable fields written to the per-model results JSONL,
# in order. ``image`` (a runtime PIL handle) is deliberately excluded.
RESULT_FIELDS: tuple[str, ...] = (
    "id",
    "image_id",
    "full_prompt",
    "question",
    "ground_truth_answer",
    "greedy_answer",
    "model_name",
    "dataset_name",
    "split",
)


def _safe(name: str) -> str:
    """Make *name* safe to use as a single path component."""
    return str(name).replace("/", "_").replace("\\", "_").strip()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def dataset_dir(dataset_key: str, root: str | Path = DEFAULT_OUTPUTS_ROOT) -> Path:
    return Path(root) / dataset_key


def images_dir(dataset_key: str, root: str | Path = DEFAULT_OUTPUTS_ROOT) -> Path:
    return dataset_dir(dataset_key, root) / "images"


def image_path(
    dataset_key: str, image_id: str, root: str | Path = DEFAULT_OUTPUTS_ROOT
) -> Path:
    return images_dir(dataset_key, root) / f"{_safe(image_id)}.png"


def results_path(
    dataset_key: str, model_name: str, root: str | Path = DEFAULT_OUTPUTS_ROOT
) -> Path:
    return dataset_dir(dataset_key, root) / f"{_safe(model_name)}.jsonl"


# ---------------------------------------------------------------------------
# Images: save once at prep time, shared by all models
# ---------------------------------------------------------------------------

def save_images(
    dataset_key: str,
    samples: list[QASample],
    root: str | Path = DEFAULT_OUTPUTS_ROOT,
) -> int:
    """Persist each sample's runtime ``image`` to the shared images folder.

    Skips samples without an image (text-only datasets) and images already on
    disk (so re-runs / other models reuse them). Returns the number written.
    """
    written = 0
    target_dir = images_dir(dataset_key, root)
    for sample in samples:
        image = sample.get("image")
        image_id = sample.get("image_id") or sample.get("id")
        if image is None or not image_id:
            continue
        path = image_path(dataset_key, str(image_id), root)
        if path.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        image.save(path)
        written += 1
    return written


def load_image(
    dataset_key: str, image_id: str, root: str | Path = DEFAULT_OUTPUTS_ROOT
):
    """Reopen a previously saved image as an RGB ``PIL.Image`` (or ``None``)."""
    path = image_path(dataset_key, image_id, root)
    if not path.is_file():
        return None
    return to_pil(str(path))


# ---------------------------------------------------------------------------
# Results JSONL: one clean per-prompt file per model
# ---------------------------------------------------------------------------

def build_result_records(samples: list[QASample]) -> list[dict[str, Any]]:
    """Project samples onto the canonical result fields (missing -> ``None``)."""
    return [{field: sample.get(field) for field in RESULT_FIELDS} for sample in samples]


def write_results_jsonl(
    dataset_key: str,
    model_name: str,
    samples: list[QASample],
    root: str | Path = DEFAULT_OUTPUTS_ROOT,
) -> Path:
    """Write ``outputs/<dataset>/<model_name>.jsonl`` and return its path."""
    path = results_path(dataset_key, model_name, root)
    write_jsonl(path, build_result_records(samples))
    return path
