"""Base types shared across all dataset adapters."""

from __future__ import annotations

from typing import Any, Sequence, TypedDict


class QASample(TypedDict, total=False):
    """Canonical in-memory QA row passed from adapters into the pipeline."""
    question: str
    reference_answer: Any
    aliases: Sequence[str] | None
    choices: list[dict[str, str]] | None
    fewshot_answer: Any
    metadata: dict[str, Any]


def as_qa_row(
    question: Any,
    reference_answer: Any,
    aliases: Sequence[str] | None = None,
    choices: list[dict[str, str]] | None = None,
    fewshot_answer: Any = None,
    metadata: dict[str, Any] | None = None,
) -> QASample:
    """Build the canonical adapter row."""
    row: QASample = {
        "question": str(question),
        "reference_answer": reference_answer,
        "aliases": aliases,
    }
    if choices:
        row["choices"] = list(choices)
    if fewshot_answer is not None:
        row["fewshot_answer"] = fewshot_answer
    if metadata:
        row["metadata"] = dict(metadata)
    return row
