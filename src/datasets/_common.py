"""Shared parsing utilities used across dataset adapters."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any


_WHITESPACE_RE = re.compile(r"\s+")
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


def normalize_text(text: Any) -> str:
    """Strip and collapse whitespace."""
    return _WHITESPACE_RE.sub(" ", str(text).strip())


def normalize_key(text: Any) -> str:
    """Lowercase, collapse whitespace, strip punctuation at edges."""
    return normalize_text(text).lower().strip(string.punctuation + " ")


def parse_text_response(response: Any) -> str:
    """Extract text content from a response that may be a dict or string."""
    if isinstance(response, dict):
        for key in ("text", "content", "response", "answer"):
            if key in response:
                return normalize_text(response[key])
        return normalize_text(str(response))
    return normalize_text(response)


def normalize_aliases(aliases: Any) -> list[str]:
    """Flatten and normalize alias lists."""
    if aliases is None:
        return []
    if isinstance(aliases, str):
        return [normalize_text(aliases)] if aliases.strip() else []
    return [normalize_text(a) for a in aliases if a and str(a).strip()]


_SCORE_STANDALONE_RE = re.compile(r"(?<!\d)[01](?!\d)")


def parse_judge_score(text: Any) -> bool | None:
    """Parse a '0' or '1' judge output, tolerating leading whitespace and trailing reasoning."""
    s = str(text).strip()
    if s in ("0", "1"):
        return s == "1"
    m = _SCORE_STANDALONE_RE.search(s)
    if m:
        return m.group(0) == "1"
    return None


parse_judge_output = parse_judge_score


def format_prompt_from_template(
    question: str,
    dataset_name: str,
    prefix: str | None = None,
) -> str:
    """Format a prompt using the template for the given dataset.
    General format : 
        [general instruction]
        
        [Fewshot prefix, if any]
        [fewshot example 1 question prefix][fewshot example 1 question]
        [fewshot example 1 answer prefix][fewshot example 1 answer]
        ...
        
        [question prefix][question]
        [answer prefix]
    """
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
    """Format a few-shot prefix using the template for the given dataset.
    General format:
        [Fewshot prefix, if any]
        [fewshot example 1 question prefix][fewshot example 1 question]
        [fewshot example 1 answer prefix][fewshot example 1 answer]
        
        [fewshot example 2 question prefix][fewshot example 2 question]
        [fewshot example 2 answer prefix][fewshot example 2 answer]
        ...        
    """
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
