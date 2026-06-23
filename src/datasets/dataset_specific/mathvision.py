"""MathVision dataset adapter (multimodal math reasoning).

HF dataset: MathLLMs/MathVision (split "test", 3040 rows).

Row schema:
  id            : str
  question      : str            (may contain "<image1>" placeholders)
  options       : list[str]      (empty for free-form, else multiple choice)
  image         : str            relative path, e.g. "images/1.jpg"
  decoded_image : PIL.Image      the actual image (HF Image feature)
  answer        : str            option letter (MC) or final value (free-form)
  solution      : str            worked solution
  level         : int
  subject       : str

The decoded image is carried inside the QASample (runtime-only ``image`` field)
and consumed by the multimodal generation backends; it is persisted once to the
shared images folder keyed by ``image_id``. ``format_prompt`` builds only the
model-agnostic text. The "<image1>" placeholder is stripped because the image is
supplied to the model out-of-band, not as an inline text marker.
"""

from __future__ import annotations

import re
from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample

_IMAGE_TAG_RE = re.compile(r"<image\d+>")

# Dataset task instruction (model-agnostic): visual math reasoning. Asking for
# step-by-step reasoning suits the benchmark and elicits chain-of-thought from
# any capable model -- it is not tied to any particular model family.
INSTRUCTION = (
    "Answer the following question about the image, "
    "explaining your reasoning step by step."
)


def _strip_image_tags(text: str) -> str:
    out = _IMAGE_TAG_RE.sub("", str(text))
    out = re.sub(r"[ \t]{2,}", " ", out)        # collapse runs of spaces
    out = re.sub(r" +([?.,;:!])", r"\1", out)    # drop space left before punctuation
    return out.strip()


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract a multimodal QA sample from a MathVision row."""
    assert "question" in example, "MathVision example is missing 'question' field"
    image = to_pil(example.get("decoded_image"))
    if image is None:
        return None

    question = str(example.get("question", "")).strip()
    options = list(example.get("options") or [])
    ground_truth_answer = str(example.get("answer", "")).strip()
    qid = str(example.get("id", "")).strip() or id

    return QASample(
        id=qid,
        image_id=qid,
        full_prompt=format_prompt(question, options),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="mathvision",
        split="test",
        image=image,
    )


# Generation functions
def format_prompt(question: str, options: list[str] | None = None) -> str:
    """Build the text prompt for a MathVision question (image added separately)."""
    question = _strip_image_tags(question)
    options = options or []

    parts: list[str] = [question]
    if options:
        lines = [f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)]
        parts.append("Options:\n" + "\n".join(lines))

    body = "\n\n".join(parts)
    return f"{INSTRUCTION}\n\n{body}"
