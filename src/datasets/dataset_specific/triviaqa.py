"""TriviaQA dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample
from src.utils.text import normalize_aliases
from src.labeling.judge import parse_judge_score


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample:
    """Extract question and answer from a TriviaQA example for dataset creation."""
    answer = example.get("answer", {})
    aliases = answer.get("aliases", []) if isinstance(answer, dict) else []
    reference_answer = answer.get("value", "") if isinstance(answer, dict) else answer
    return QASample(
        question=example.get("question", ""),
        reference_answer=reference_answer,
        aliases=normalize_aliases(aliases),
        id=str(example.get("question_id", None))
    )


# Generation functions
def format_prompt(
    qa_sample: QASample,
    fewshot_prefix: str | None = None,
) -> str:
    """Format a prompt for TriviaQA."""
    if fewshot_prefix:
        return fewshot_prefix + "Question: " + qa_sample["question"] + "\nAnswer:"
    return "Question: " + qa_sample["question"] + "\nAnswer:"


def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for TriviaQA."""
    prefix = ""
    for example in few_shot_examples or []:
        prefix += f"Question: {example['question']}\nAnswer: {example['reference_answer']}\n\n"
    return prefix


# Labeling functions
def build_judge_prompt(record: QASample) -> str:
    """Build a prompt for LLM judging of a TriviaQA record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "reference_answer" in record, "Record must contain 'reference_answer' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key with model response"

    ref = record.get("reference_answer", "")
    aliases = record.get("aliases")
    aliases = aliases if isinstance(aliases, list) and aliases else [ref]
    refs = ", ".join(a for a in aliases if a)

    prompt = "Task : You evaluate whether the model response correctly answers the question.\n"
    prompt += "Accept conversational hedges if the core answer matches any reference alias.\n"
    prompt += f"Question: {record['question']}\n"
    prompt += f"Acceptable response(s): {refs}\n"
    prompt += f"Model response: {record['raw_answer']}\n"
    prompt += "Answer only 0 (incorrect or uncommitted) or 1 (correct).\n"
    prompt += "Output a single digit, nothing else.\n"
    prompt += "Score:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a TriviaQA record into a label."""
    return parse_judge_score(text)
