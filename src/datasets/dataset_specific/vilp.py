"""ViLP visual-language-priors probe adapter (multimodal).

HF dataset: ViLP/ViLP (train split, 300 rows).

Row schema:
  question : str           one question shared by all three variants
  image1   : Image         variant-1 image
  answer1  : str           variant-1 answer
  image2 / answer2         variant-2 image + answer
  image3 / answer3         variant-3 image + answer

ViLP pairs one question with three different (image, answer) variants to test
whether a model actually looks at the image instead of relying on language priors.
``extract_qa_sample`` therefore fans each row out into up to three independent
QASamples (one per image), each carrying its own reference answer and its own
``id`` / ``image_id`` (``<base>_v<n>``) so the three images are stored separately.
The dataloader flattens the returned list.
"""

from __future__ import annotations

from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample

_VARIANTS = (("image1", "answer1"), ("image2", "answer2"), ("image3", "answer3"))


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> list[QASample]:
    """Fan one ViLP row out into one QASample per (image, answer) variant."""
    question = str(example.get("question", "")).strip()
    if not question:
        return []
    base = str(example.get("id", "")).strip() or id
    samples: list[QASample] = []
    for n, (img_key, ans_key) in enumerate(_VARIANTS, start=1):
        image = to_pil(example.get(img_key))
        ground_truth_answer = str(example.get(ans_key, "")).strip()
        if image is None or not ground_truth_answer:
            continue
        variant_id = f"{base}_v{n}"
        samples.append(QASample(
            id=variant_id,
            image_id=variant_id,
            full_prompt=format_prompt(question),
            question=question,
            ground_truth_answer=ground_truth_answer,
            model_name=None,
            dataset_name="vilp",
            split="train",
            image=image,
        ))
    return samples


# Generation functions
def format_prompt(question: str) -> str:
    """Build the text prompt for a ViLP question (image added separately)."""
    instruction = (
        "Look carefully at the image and answer the question based on what you "
        "actually see. Give a short, direct answer."
    )
    return f"{instruction}\n\nQuestion: {question}\nAnswer:"
