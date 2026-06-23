"""DrivingVQA driving-theory visual-question-answering adapter (multimodal).

Local dataset: ``datasets/DrivingVQA/test.jsonl`` (789 rows), built from the raw
``test.json`` by ``scripts/prepare_drivingvqa.py``.

Row schema (one record per line):
  id               : str            sample id
  image_path       : str            repo-relative path to the dashcam image
  questions        : str | list[str] one question, or two sub-questions sharing the image
  possible_answers : dict           {"A": text, "B": text, ...} lettered options
  true_answers     : list[str]      one or more correct option letters
  explanation      : str            human rationale (not shown to the model)

DrivingVQA is multiple-choice where one *or more* options may be correct (and an
image may bundle two sub-questions sharing the same option pool). The image rides
inside the QASample (runtime-only ``image`` field) and is persisted once to the
shared images folder keyed by ``image_id``; ``format_prompt`` builds only the
text. ``ground_truth_answer`` is the correct option letters paired with their
text so the judge has an unambiguous target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample

INSTRUCTION = (
    "You are given an image from a driving-theory exam. Answer the "
    "multiple-choice question below. One or more options may be correct -- "
    "give every correct option letter."
)


def _normalise_questions(questions: Any) -> list[str]:
    """Coerce the ``questions`` field (str or list) into a list of question strings."""
    if isinstance(questions, list):
        return [str(q).strip() for q in questions if str(q).strip()]
    text = str(questions or "").strip()
    return [text] if text else []


def _format_ground_truth(possible_answers: dict[str, Any], true_answers: list[str]) -> str:
    """Render the correct answers as ``letter) text`` pairs joined by ``; ``."""
    parts = []
    for letter in true_answers:
        text = str(possible_answers.get(letter, "")).strip()
        parts.append(f"{letter}) {text}" if text else str(letter))
    return "; ".join(parts)


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract image + question(s) + correct options from a DrivingVQA row."""
    image = to_pil(example.get("image_path"))
    questions = _normalise_questions(example.get("questions"))
    possible_answers = example.get("possible_answers") or {}
    true_answers = [str(a).strip() for a in (example.get("true_answers") or []) if str(a).strip()]
    if image is None or not questions or not possible_answers or not true_answers:
        return None

    qid = str(example.get("id", "")).strip() or id
    # image_id is the image stem so rows sharing an image store it once.
    image_id = Path(str(example.get("image_path", ""))).stem or qid

    return QASample(
        id=qid,
        image_id=image_id,
        full_prompt=format_prompt(questions, possible_answers),
        question="\n".join(questions),
        ground_truth_answer=_format_ground_truth(possible_answers, true_answers),
        model_name=None,
        dataset_name="drivingvqa",
        split="test",
        image=image,
    )


# Generation functions
def format_prompt(questions: list[str] | str, possible_answers: dict[str, Any]) -> str:
    """Build the text prompt for a DrivingVQA question (image added separately)."""
    questions = _normalise_questions(questions)
    if len(questions) == 1:
        q_block = f"Question: {questions[0]}"
    else:
        q_block = "\n".join(f"Question {i}: {q}" for i, q in enumerate(questions, start=1))

    options = "\n".join(
        f"({letter}) {str(text).strip()}" for letter, text in possible_answers.items()
    )
    return f"{INSTRUCTION}\n\n{q_block}\n\nOptions:\n{options}\nAnswer:"
