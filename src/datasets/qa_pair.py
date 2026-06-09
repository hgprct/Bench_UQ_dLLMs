"""QA pair definition and construction utilities.

All fields are optional. Each dataset populates the fields it supports
and consumers check for what they need.
"""

from __future__ import annotations

from typing import Any, Sequence, TypedDict


class QASample(TypedDict, total=False):
    """Canonical in-memory QA row passed from adapters into the pipeline."""
    question: str
    reference_answer: Any
    aliases: Sequence[str] | None
    choices: list[dict[str, str]] | None
    fewshot_answer: Any
    source_document: str | None
    context: str | None
    task_type: str
    id: str | int
    level: str
    type: str
    supporting_facts: Any
    answerable: bool
    question_decomposition: Any
    source_language: str
    target_language: str
    language_pair: str


def as_qa_row(
    question: Any,
    reference_answer: Any,
    aliases: Sequence[str] | None = None,
    choices: list[dict[str, str]] | None = None,
    fewshot_answer: Any = None,
    **extra: Any,
) -> QASample:
    """Build a QA row with core fields plus any extra dataset-specific fields."""
    row: QASample = {
        "question": str(question),
        "reference_answer": reference_answer,
        "aliases": aliases,
    }
    if choices:
        row["choices"] = list(choices)
    if fewshot_answer is not None:
        row["fewshot_answer"] = fewshot_answer
    for k, v in extra.items():
        if v is not None:
            row[k] = v
    return row
