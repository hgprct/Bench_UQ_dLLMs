"""Template I/O and prompt formatting utilities for dataset adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TEMPLATES_DIR = Path(__file__).parent
_PROMPT_TEMPLATES: dict | None = None
_JUDGE_TEMPLATES: dict | None = None


def get_prompt_template(dataset_name: str) -> dict[str, str]:
    global _PROMPT_TEMPLATES
    if _PROMPT_TEMPLATES is None:
        with open(_TEMPLATES_DIR / "prompt_templates.json") as f:
            _PROMPT_TEMPLATES = json.load(f)
    if dataset_name not in _PROMPT_TEMPLATES:
        raise ValueError(f"No prompt template for '{dataset_name}'. Valid: {sorted(_PROMPT_TEMPLATES)}")
    return _PROMPT_TEMPLATES[dataset_name]


def get_judge_template(dataset_name: str) -> dict[str, str | None]:
    global _JUDGE_TEMPLATES
    if _JUDGE_TEMPLATES is None:
        with open(_TEMPLATES_DIR / "judge_templates.json") as f:
            _JUDGE_TEMPLATES = json.load(f)
    if dataset_name not in _JUDGE_TEMPLATES:
        raise ValueError(f"No judge template for '{dataset_name}'. Valid: {sorted(_JUDGE_TEMPLATES)}")
    return _JUDGE_TEMPLATES[dataset_name]


def _extract_candidate(record: dict[str, Any]) -> str:
    final = record.get("final", {})
    return str(final.get("answer", final.get("response", "")) or "")


def build_judge_prompt_from_template(
    record: dict[str, Any],
    dataset_name: str,
    *,
    references: str | None = None,
) -> str:
    t = get_judge_template(dataset_name)
    source = record.get("question", "")
    candidate = _extract_candidate(record)

    sep = t.get("fields_separator", "\n\n")
    fields = [f"{t['source_label']}: {source}"]
    if t["reference_label"] is not None and references is not None:
        fields.append(f"{t['reference_label']}: {references}")
    fields.append(f"{t['candidate_label']}: {candidate}")

    return (
        t["preamble"] + sep
        + "\n".join(fields)
        + "\n\n" + t["instruction"] + "\n" + t["score_label"]
    )


def format_prompt_from_template(
    question: str,
    dataset_name: str,
    prefix: str | None = None,
) -> str:
    t = get_prompt_template(dataset_name)
    parts = []
    if t["general_instruction"]:
        parts.append(t["general_instruction"])
    if prefix:
        parts.append(str(prefix).strip())
    parts.append(f"{t['question_prefix']}{str(question).strip()}\n{t['answer_prefix']}")
    return "\n\n".join(parts).strip()


def few_shot_prefix_from_template(
    few_shot_examples: list[dict[str, Any]] | None,
    dataset_name: str,
) -> str:
    if not few_shot_examples:
        return ""
    t = get_prompt_template(dataset_name)
    answer_sep = t.get("fewshot_answer_sep", "")
    intro_sep = t.get("fewshot_intro_sep", "\n\n")
    examples = "\n\n".join(
        f"{t['question_prefix']}{str(example.get('question', '')).strip()}\n"
        f"{t['answer_prefix']}{answer_sep}{str(example.get('answer', '')).strip()}"
        for example in few_shot_examples
    )
    if t["examples_introduction"]:
        return f"{t['examples_introduction']}{intro_sep}{examples}"
    return examples
