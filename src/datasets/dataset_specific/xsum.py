"""XSum extreme summarization dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract document and summary from an XSum example for dataset creation."""
    assert "document" in example, "XSum example is missing 'document' field"
    assert "summary" in example, "XSum example is missing 'summary' field"
    source = str(example.get("document", "")).strip()
    summary = str(example.get("summary", "")).strip()

    return QASample(
        question=source,
        reference_answer=summary,
        id=str(example.get("id", None)),
    )


# Generation functions
def format_prompt(
    qa_sample: QASample,
    fewshot_prefix: str | None = None,
) -> str:
    """Format a prompt for XSum summarization."""
    instruction = "Provide a short one-sentence summary for the following text."
    parts = [instruction]
    if fewshot_prefix:
        parts.append(str(fewshot_prefix).strip())
    parts.append(
        f"Here's the text:\n{qa_sample['question']}\n"
        f"(End of text).\n\nSummary:"
    )
    return "\n\n".join(parts).strip()


def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for XSum."""
    if not few_shot_examples:
        return ""
    examples = []
    for ex in few_shot_examples:
        q = ex["question"]
        a = ex["reference_answer"]
        examples.append(
            f"Here's the text:\n{q}\n(End of text).\n\nSummary: {a}"
        )
    return "Here are some examples:\n" + "\n\n".join(examples)


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of an XSum record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key with model response"

    prompt = "Task : You evaluate the consistency of a summary with the source document.\n\n"
    prompt += f"Document: \n{record['question']}\n\n"
    prompt += f"Summary: {record['raw_answer']}\n\n"
    prompt += "No hallucinations or contradictions allowed. "
    prompt += "Answer only 0 (inconsistent) or 1 (consistent).\n"
    prompt += "Output a single digit, nothing else.\nScore:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for an XSum record into a label."""
    return parse_judge_score(text)
