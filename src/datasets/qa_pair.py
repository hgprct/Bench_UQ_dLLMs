"""QA pair definition.

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
    raw_reference_answer: Any 
    raw_answer: Any
    parsed_answer: Any
    label: bool | None
    source_documents: list[dict[str, str]] | None
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


