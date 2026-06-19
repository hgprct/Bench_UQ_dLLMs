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
QASamples (one per image), each carrying its own reference answer. The dataloader
flattens the returned list.
"""

from __future__ import annotations

from typing import Any

from src.datasets.image_utils import to_pil
from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score

_VARIANTS = (("image1", "answer1"), ("image2", "answer2"), ("image3", "answer3"))


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> list[QASample]:
    """Fan one ViLP row out into one QASample per (image, answer) variant."""
    question = str(example.get("question", "")).strip()
    if not question:
        return []
    samples: list[QASample] = []
    for n, (img_key, ans_key) in enumerate(_VARIANTS, start=1):
        image = to_pil(example.get(img_key))
        answer = str(example.get(ans_key, "")).strip()
        if image is None or not answer:
            continue
        samples.append(QASample(
            question=question,
            reference_answer=answer,
            image=image,
            id=f"{example.get('id', '')}_v{n}".strip("_"),
            subject=f"variant_{n}",
            task_type="multimodal_vilp",
        ))
    return samples


# Generation functions
def format_prompt(qa_sample: QASample, fewshot_prefix: str | None = None) -> str:
    """Build the text prompt for a ViLP question (image added separately)."""
    instruction = (
        "Look carefully at the image and answer the question based on what you "
        "actually see. Give a short, direct answer."
    )
    prefix = fewshot_prefix or ""
    return f"{instruction}\n\n{prefix}Question: {qa_sample['question']}\nAnswer:"


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of a ViLP record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key"
    prompt = "Task : You evaluate whether the model's answer matches the "
    prompt += "reference answer for this image.\n"
    prompt += "Accept paraphrases and equivalent phrasings.\n"
    prompt += f"Question: {record['question']}\n"
    prompt += f"Reference answer: {record.get('reference_answer', '')}\n"
    prompt += f"Model response: {record['raw_answer']}\n"
    prompt += "Answer only 0 (incorrect) or 1 (correct).\n"
    prompt += "Output a single digit, nothing else.\nScore:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a ViLP record into a label."""
    return parse_judge_score(text)
