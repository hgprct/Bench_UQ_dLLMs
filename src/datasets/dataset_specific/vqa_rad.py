"""VQA-RAD radiology visual-question-answering adapter (multimodal).

HF dataset: flaviagiammarino/vqa-rad (train / test).

Row schema:
  image    : Image    the radiology image (PIL)
  question : str      the question about the image
  answer   : str      short free-form answer (often yes/no or a single phrase)

The image rides inside the QASample (``image`` field) and is consumed by the
multimodal generation path; ``format_prompt`` builds only the text. Answers are
short and free-form, so an LLM judge labels them.
"""

from __future__ import annotations

from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract image + question + answer from a VQA-RAD row."""
    image = to_pil(example.get("image"))
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    if image is None or not question:
        return None
    return QASample(
        question=question,
        reference_answer=answer,
        image=image,
        task_type="multimodal_vqa",
    )


# Generation functions
def format_prompt(qa_sample: QASample, fewshot_prefix: str | None = None) -> str:
    """Build the text prompt for a VQA-RAD question (image added separately)."""
    instruction = (
        "Answer the following question about the medical image. "
        "Give a short, direct answer."
    )
    prefix = fewshot_prefix or ""
    return f"{instruction}\n\n{prefix}Question: {qa_sample['question']}\nAnswer:"


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of a VQA-RAD record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key"
    prompt = "Task : You evaluate whether the model's answer to a medical-image "
    prompt += "question matches the reference answer.\n"
    prompt += "Accept paraphrases and equivalent yes/no phrasings.\n"
    prompt += f"Question: {record['question']}\n"
    prompt += f"Reference answer: {record.get('reference_answer', '')}\n"
    prompt += f"Model response: {record['raw_answer']}\n"
    prompt += "Answer only 0 (incorrect) or 1 (correct).\n"
    prompt += "Output a single digit, nothing else.\nScore:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a VQA-RAD record into a label."""
    return parse_judge_score(text)
