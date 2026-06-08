"""TriviaQA dataset adapter."""

from __future__ import annotations

import re
import string
from typing import Any, Sequence

from src.datasets._base import QASample, as_qa_row
from src.datasets._common import build_judge_prompt_from_template, few_shot_prefix_from_template, format_prompt_from_template, normalize_aliases, normalize_key, normalize_text, parse_judge_output, parse_text_response

HF_NAME = "mandarjoshi/trivia_qa"
DEFAULT_CONFIG_NAME = "rc"
DEFAULT_SPLIT = "validation"

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation + "''´`")
_SHORT_ANSWER_PATTERNS = (
    re.compile(r"^the answer is\s+(.+)$", flags=re.IGNORECASE),
    re.compile(r"^answer:\s*(.+)$", flags=re.IGNORECASE),
    re.compile(r"^final answer:\s*(.+)$", flags=re.IGNORECASE),
)


def load_dataset(
    split: str = DEFAULT_SPLIT,
    dataset_config_name: str = DEFAULT_CONFIG_NAME,
    token: str | None = None,
):
    from datasets import load_dataset
    if dataset_config_name is None:
        return load_dataset(HF_NAME, split=split, token=token)
    return load_dataset(HF_NAME, dataset_config_name, split=split, token=token)


def extract_question_and_answer(example: dict[str, Any]) -> QASample:
    answer = example.get("answer", {})
    aliases = answer.get("aliases", []) if isinstance(answer, dict) else []
    reference_answer = answer.get("value", "") if isinstance(answer, dict) else answer
    return as_qa_row(example.get("question", ""), reference_answer, normalize_aliases(aliases))


def format_prompt(
    question: str,
    prefix: str | None = None,
    choices: Sequence[dict[str, str]] | None = None,
) -> str:
    del choices
    return format_prompt_from_template(question, "triviaqa", prefix=prefix)


def few_shot_prefix(few_shot_examples: list[dict[str, Any]] | None = None) -> str:
    return few_shot_prefix_from_template(few_shot_examples, "triviaqa")


def normalize_answer(text: Any) -> str:
    """TriviaQA/SQuAD-style normalization for exact-match checks."""
    text = str(text).replace("_", " ").lower()
    text = "".join(" " if char in _PUNCTUATION else char for char in text)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def extract_short_answer(text: Any) -> str:
    text = str(text).strip()
    for pattern in _SHORT_ANSWER_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).strip()
    return text


def cluster_key(response: Any, sample: QASample | None = None) -> str:
    del sample
    return normalize_key(parse_text_response(response))


def parse_response(response: Any) -> str:
    return parse_text_response(response)


def build_judge_prompt(record: dict[str, Any]) -> str:
    ref = record.get("reference_answer", "")
    aliases = record.get("answer_aliases", record.get("aliases"))
    gold_answers = aliases if isinstance(aliases, list) and aliases else [ref]
    refs = ", ".join(a for a in gold_answers if a)
    return build_judge_prompt_from_template(record, "triviaqa", references=refs)
