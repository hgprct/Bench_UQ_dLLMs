"""GSM8K dataset adapter."""

from __future__ import annotations
from typing import Any

from src.datasets.qa_pair import QASample

# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract question and answer from a GSM8K example for dataset creation."""
    assert "question" in example, "GSM8K example is missing 'question' field"
    assert "answer" in example, "GSM8K example is missing 'answer' field"
    question = str(example.get("question", "")).strip()
    ground_truth_answer = str(example.get("answer", "")).strip()

    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="gsm8k",
        split="test"
    )


# Generation functions
def format_prompt(question: str) -> str:
    """Format a prompt for GSM8K math problem solving."""
    prompt = "Given the following question, reason and give a final answer to the question. Your response should end with \"#### [answer]\" where [answer] is the response to the question.\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt