"""DriveBench arena driving-scene VQA adapter (multimodal).

HF dataset: drive-bench/arena (test split, 1461 rows).

Row schema:
  scene_token / frame_token : str
  question_type             : str   task / category
  question                  : str
  answer                    : str   ground-truth
  tag                       : list[int]
  image_path                : struct of 6 camera-view path strings:
        CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
        CAM_BACK,  CAM_BACK_LEFT,  CAM_BACK_RIGHT

CAREFUL-ATTENTION ITEM: this dataset ships image *paths* (into nuScenes), not
pixels. To run it you must have the nuScenes images locally and point
``DRIVE_BENCH_IMAGE_ROOT`` at the directory the paths are relative to. Rows whose
front-camera image cannot be resolved are skipped (with a one-time warning). We
use the single CAM_FRONT view (the diffusion VLMs here take one image).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score

_PRIMARY_CAM = "CAM_FRONT"
_warned = False


def _image_root() -> Path | None:
    root = os.environ.get("DRIVE_BENCH_IMAGE_ROOT", "").strip()
    return Path(root) if root else None


def _resolve_front_image(image_path: Any):
    """Resolve the CAM_FRONT path against DRIVE_BENCH_IMAGE_ROOT -> PIL or None."""
    global _warned
    if not isinstance(image_path, dict):
        return None
    rel = image_path.get(_PRIMARY_CAM)
    if not rel:
        return None
    root = _image_root()
    candidate = Path(rel) if root is None else root / rel
    if not candidate.is_file():
        if not _warned:
            print(
                f"[drive_bench] image not found: {candidate}. Set "
                "DRIVE_BENCH_IMAGE_ROOT to the nuScenes image root; "
                "rows without resolvable images are skipped."
            )
            _warned = True
        return None
    return to_pil(str(candidate))


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract front-camera image + question + answer from a DriveBench row."""
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    image = _resolve_front_image(example.get("image_path"))
    if image is None or not question:
        return None
    return QASample(
        question=question,
        reference_answer=answer,
        image=image,
        subject=str(example.get("question_type", "")).strip(),
        id=str(example.get("frame_token", "")),
        task_type="multimodal_driving",
    )


# Generation functions
def format_prompt(qa_sample: QASample, fewshot_prefix: str | None = None) -> str:
    """Build the text prompt for a DriveBench question (image added separately)."""
    instruction = (
        "You are given a front-camera image from a driving scene. "
        "Answer the following question about the scene."
    )
    prefix = fewshot_prefix or ""
    return f"{instruction}\n\n{prefix}Question: {qa_sample['question']}\nAnswer:"


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of a DriveBench record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key"
    prompt = "Task : You evaluate whether the model's answer about a driving "
    prompt += "scene matches the reference answer.\n"
    prompt += "Accept paraphrases and equivalent descriptions.\n"
    prompt += f"Question: {record['question']}\n"
    prompt += f"Reference answer: {record.get('reference_answer', '')}\n"
    prompt += f"Model response: {record['raw_answer']}\n"
    prompt += "Answer only 0 (incorrect) or 1 (correct).\n"
    prompt += "Output a single digit, nothing else.\nScore:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a DriveBench record into a label."""
    return parse_judge_score(text)
