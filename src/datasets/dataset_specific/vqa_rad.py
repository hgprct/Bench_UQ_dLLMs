"""VQA-RAD radiology visual-question-answering adapter (multimodal).

HF dataset: flaviagiammarino/vqa-rad (train / test).

Row schema:
  image    : Image    the radiology image (PIL)
  question : str      the question about the image
  answer   : str      short free-form answer (often yes/no or a single phrase)

The image rides inside the QASample (runtime-only ``image`` field) and is consumed
by the multimodal generation path; it is persisted once to the shared images
folder keyed by ``image_id``. ``format_prompt`` builds only the text.
"""

from __future__ import annotations

from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract image + question + answer from a VQA-RAD row."""
    image = to_pil(example.get("image"))
    question = str(example.get("question", "")).strip()
    ground_truth_answer = str(example.get("answer", "")).strip()
    if image is None or not question:
        return None

    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="vqa_rad",
        split="test",
        image=image,
    )


# Generation functions
def format_prompt(question: str) -> str:
    """Build the text prompt for a VQA-RAD question (image added separately)."""
    instruction = (
        "Answer the following question about the medical image. "
        "Give a short, direct answer."
    )
    return f"{instruction}\n\nQuestion: {question}\nAnswer:"
