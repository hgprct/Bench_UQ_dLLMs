"""GSM8K dataset adapter."""

from __future__ import annotations

import re
from typing import Any

from src.datasets.qa_pair import QASample


_GSM8K_FINAL_MARKER_RE = re.compile(r"####\s*([^\r\n]*)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract question and answer from a GSM8K example for dataset creation."""
    assert "question" in example, "GSM8K example is missing 'question' field"
    assert "answer" in example, "GSM8K example is missing 'answer' field"
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    parsed_answer = _extract_marked_number(answer)

    return QASample(
        question=question,
        reference_answer=parsed_answer,
        raw_reference_answer=answer,
    )

# Generation functions
def format_prompt(
    qa_sample: QASample,
    fewshot_prefix: str | None = None,
) -> str:
    """Format a prompt for GSM8K math problem solving."""
    prompt = "Given the following question, reason and give a final answer to the question. Your response should end with \"#### [answer]\" where [answer] is the response to the question.\n\n"
    prompt += fewshot_prefix or ""
    prompt += f"Question: {qa_sample['question']}\nAnswer:"
    return prompt

def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for GSM8K."""
    if not few_shot_examples:
        return ""
    examples = []
    for ex in few_shot_examples:
        q = ex["question"]
        a = ex["raw_reference_answer"]
        examples.append(
            f"Question: {q}\nAnswer: {a}"
        )
    return "Here are a few examples of questions and answers:\n" + "\n\n".join(examples) + "\n\n"

def parse_answer(raw_answer: str) -> str:
    return _extract_marked_number(raw_answer)

def _extract_marked_number(text: Any) -> str:
    markers = list(_GSM8K_FINAL_MARKER_RE.finditer(str(text)))
    if not markers:
        return ""
    answer_line = markers[-1].group(1).replace(",", "")
    number = _NUMBER_RE.search(answer_line)
    return number.group(0) if number else ""