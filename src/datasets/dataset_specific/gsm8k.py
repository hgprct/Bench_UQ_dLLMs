"""GSM8K dataset adapter."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

from src.datasets.qa_pair import QASample, as_qa_row
from src.datasets._templates import few_shot_prefix_from_template, format_prompt_from_template
from src.utils.text import normalize_key, normalize_text

HF_NAME = "openai/gsm8k"
DEFAULT_CONFIG_NAME = "main"
DEFAULT_SPLIT = "test"
DEFAULT_LABEL_METHOD = "exact_match"
LABEL_GPU_MODE = "cpu"
FEWSHOT_K = 4

_GSM8K_FINAL_MARKER_RE = re.compile(r"####\s*([^\r\n]*)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    chars = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        index += 1
    return None


def parse_math_response(response: Any) -> str:
    text = str(response).strip()
    marker = re.search(r"####\s*(.*)$", text, flags=re.DOTALL)
    if marker:
        text = marker.group(1).strip()
    else:
        boxed = boxed_content(text)
        if boxed is not None:
            text = boxed
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                text = lines[-1]
    text = re.sub(r"^(?:answer|final answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip(".")


def extract_last_number(text: Any) -> str:
    normalized = str(parse_math_response(text)).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    return matches[-1] if matches else ""


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
    return _extract_marked_number(answer_text) or extract_last_number(answer_text)


def parse_response(response: Any) -> str:
    return _extract_marked_number(response) or extract_last_number(response)


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
