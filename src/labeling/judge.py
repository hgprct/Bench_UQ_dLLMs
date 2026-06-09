"""Generic LLM judge labeling with retry logic and score parsing."""

from __future__ import annotations

import re
from typing import Any, Callable

_SCORE_STANDALONE_RE = re.compile(r"(?<!\d)[01](?!\d)")


def parse_judge_score(text: Any) -> bool | None:
    """Parse a '0' or '1' judge output, tolerating whitespace and trailing reasoning."""
    s = str(text).strip()
    if s in ("0", "1"):
        return s == "1"
    m = _SCORE_STANDALONE_RE.search(s)
    if m:
        return m.group(0) == "1"
    return None


def response_text(record: dict[str, Any]) -> str:
    """Extract candidate response text from a record."""
    raw = record.get("raw_answer")
    if raw is not None:
        return str(raw).strip()
    return ""


def judge_label_records_batch(
    records: list[dict[str, Any]],
    evaluator: Any,
    build_judge_prompt: Callable[[dict[str, Any]], str],
    parse_judge_output: Callable[[Any], bool | None],
) -> tuple[list[bool | None], set[int]]:
    """Label records using an LLM judge with one retry for ambiguous outputs."""
    assert evaluator is not None, "Evaluator model must be provided for judge labeling"
    labels: list[bool | None] = [None] * len(records)
    judge_indices: list[int] = []
    for i, record in enumerate(records):
        if response_text(record) == "":
            labels[i] = False
        else:
            judge_indices.append(i)

    if not judge_indices:
        return labels, set()

    prompts = [build_judge_prompt(records[i]) for i in judge_indices]
    raw_outputs = evaluator.predict_batch(prompts, temperature=0.0, max_tokens=16)
    for idx, out in zip(judge_indices, raw_outputs):
        labels[idx] = parse_judge_output(out)

    retry_indices = [idx for idx in judge_indices if labels[idx] is None]
    retried: set[int] = set(retry_indices)
    if retry_indices:
        retry_prompts = [build_judge_prompt(records[i]) for i in retry_indices]
        retry_outputs = evaluator.predict_batch(retry_prompts, temperature=0.0, max_tokens=16)
        for idx, out in zip(retry_indices, retry_outputs):
            labels[idx] = parse_judge_output(out)

    return labels, retried
