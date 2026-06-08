"""GSM8K dataset adapter."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

from src.datasets._base import QASample, as_qa_row
from src.datasets._common import few_shot_prefix_from_template, format_prompt_from_template, normalize_key, normalize_text

HF_NAME = "openai/gsm8k"
DEFAULT_CONFIG_NAME = "main"
DEFAULT_SPLIT = "test"

_GSM8K_FINAL_MARKER_RE = re.compile(r"####\s*([^\r\n]*)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_dataset(
    split: str = DEFAULT_SPLIT,
    dataset_config_name: str | None = DEFAULT_CONFIG_NAME,
    token: str | None = None,
):
    from datasets import load_dataset
    if dataset_config_name is None:
        return load_dataset(HF_NAME, split=split, token=token)
    return load_dataset(HF_NAME, dataset_config_name, split=split, token=token)


def extract_question_and_answer(example: dict[str, Any]) -> QASample:
    question = example.get("question", "")
    answer = example.get("answer", "")
    return as_qa_row(
        question,
        extract_final_answer(answer),
        fewshot_answer=str(answer).strip(),
    )


def format_prompt(
    question: str,
    prefix: str | None = None,
    choices: Sequence[dict[str, str]] | None = None,
) -> str:
    del choices
    return format_prompt_from_template(question, "gsm8k", prefix=prefix)


def few_shot_prefix(few_shot_examples: list[dict[str, Any]] | None = None) -> str:
    return few_shot_prefix_from_template(few_shot_examples, "gsm8k")


def extract_final_answer(answer_text: Any) -> str:
    return _extract_marked_number(answer_text) or _extract_last_number(answer_text)


def parse_response(response: Any) -> str:
    return _extract_marked_number(response) or _extract_last_number(response)


def extract_short_answer(response: Any) -> str:
    return parse_response(response)


def normalize_answer(answer: Any) -> str:
    return _canonical_number(parse_response(answer))


def cluster_key(response: Any, sample: QASample | None = None) -> str:
    del sample
    parsed = normalize_answer(response)
    return normalize_key(parsed) if parsed else normalize_text(response)


def _extract_marked_number(text: Any) -> str:
    markers = list(_GSM8K_FINAL_MARKER_RE.finditer(str(text)))
    if not markers:
        return ""
    answer_line = markers[-1].group(1).replace(",", "")
    number = _NUMBER_RE.search(answer_line)
    return number.group(0) if number else ""


def _extract_last_number(text: Any) -> str:
    from src.datasets._common_math import extract_last_number
    return extract_last_number(text)


def _canonical_number(number: Any) -> str:
    text = str(number).strip().replace(",", "")
    if not text:
        return ""
    try:
        value = Decimal(text)
    except InvalidOperation:
        return normalize_key(text)
    formatted = format(value.normalize(), "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return "0" if formatted in {"", "-0"} else formatted


def exact_match_correct(prediction: str, gold_answers: Sequence[Any]) -> bool:
    pred_num = normalize_answer(prediction)
    return any(
        bool(pred_num and pred_num == normalize_answer(str(ref)))
        for ref in gold_answers
    )
