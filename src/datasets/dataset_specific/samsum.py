"""SamSum dialogue summarization dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract dialogue and summary from a SamSum example for dataset creation."""
    assert "dialogue" in example, "SamSum example is missing 'dialogue' field"
    assert "summary" in example, "SamSum example is missing 'summary' field"
    source = str(example.get("dialogue", "")).strip()
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
    """Format a prompt for SamSum summarization."""
    instruction = (
        "Provide a summary for the following dialogue. The summary should "
        "(1) be rather short, (2) extract important pieces of information, "
        "(3) include names of interlocutors, (4) be written in the third person."
    )
    parts = [instruction]
    if fewshot_prefix:
        parts.append(str(fewshot_prefix).strip())
    parts.append(
        f"Here's the dialogue:\n{qa_sample['question']}\n"
        f"(End of dialogue).\n\nSummary:"
    )
    return "\n\n".join(parts).strip()


def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for SamSum."""
    if not few_shot_examples:
        return ""
    examples = []
    for ex in few_shot_examples:
        q = ex["question"]
        a = ex["reference_answer"]
        examples.append(
            f"Here's the dialogue:\n{q}\n(End of dialogue).\n\nSummary: {a}"
        )
    return "Here are some examples:\n" + "\n\n".join(examples)


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of a SamSum record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key with model response"

    prompt = "Task : You evaluate the consistency of a summary with the source dialogue.\n\n"
    prompt += f"Dialogue: \n{record['question']}\n\n"
    prompt += f"Summary: {record['raw_answer']}\n\n"
    prompt += "No hallucinations or contradictions allowed. "
    prompt += "Answer only 0 (inconsistent) or 1 (consistent).\n"
    prompt += "Output a single digit, nothing else.\nScore:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a SamSum record into a label."""
    return parse_judge_score(text)
