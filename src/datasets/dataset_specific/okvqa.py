"""OK-VQA outside-knowledge visual-question-answering adapter (multimodal).

Local dataset: ``datasets/OKVQA/test.jsonl`` (COCO val2014 split, 5046 rows),
built by ``scripts/prepare_okvqa.py`` from the OK-VQA questions + annotations.

Row schema (one record per line):
  id         : str        question id
  image_id   : str        COCO image id (shared across that image's questions)
  image_path : str        repo-relative path to the COCO val2014 image
  question   : str        the open-ended question
  answer     : str        majority of the ten human answers (canonical GT)
  answers    : list[str]  distinct human answers (any may be acceptable)

The image rides inside the QASample (runtime-only ``image`` field) and is
persisted once to the shared images folder keyed by ``image_id`` (so questions
sharing an image store it once); ``format_prompt`` builds only the text.
"""

from __future__ import annotations

from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract image + question + answer from an OK-VQA row."""
    image = to_pil(example.get("image_path"))
    question = str(example.get("question", "")).strip()
    ground_truth_answer = str(example.get("answer", "")).strip()
    if image is None or not question:
        return None

    qid = str(example.get("id", "")).strip() or id
    image_id = str(example.get("image_id", "")).strip() or qid

    return QASample(
        id=qid,
        image_id=image_id,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="okvqa",
        split="test",
        image=image,
    )


# Generation functions
def format_prompt(question: str) -> str:
    """Build the text prompt for an OK-VQA question (image added separately)."""
    instruction = (
        "Look carefully at the image and answer the question. The question may "
        "require outside knowledge. Give a short, direct answer."
    )
    return f"{instruction}\n\nQuestion: {question}\nAnswer:"
