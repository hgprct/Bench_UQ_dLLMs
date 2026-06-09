"""Shared WMT translation adapter logic for all language pairs."""

from __future__ import annotations

import re
from typing import Any, Sequence

from src.datasets.qa_pair import QASample, as_qa_row
from src.utils.text import normalize_key

_ANSWER_PREFIX_RE = re.compile(
    r"^(?:translation|english|answer|final answer)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
_ECHOED_PROMPT_RE = re.compile(
    r"(?:^|\n)\s*(?:source|german|french)\s*:\s*",
    flags=re.IGNORECASE,
)


def extract_translation_pair(
    example: dict[str, Any],
    *,
    source_language: str,
    target_language: str = "en",
) -> QASample:
    translation = example.get("translation", example)
    if not isinstance(translation, dict):
        raise ValueError("WMT examples must expose a translation dictionary")
    if source_language not in translation:
        raise ValueError(f"WMT example is missing source language '{source_language}'")
    if target_language not in translation:
        raise ValueError(f"WMT example is missing target language '{target_language}'")

    extra: dict[str, Any] = {
        "source_language": source_language,
        "target_language": target_language,
        "language_pair": f"{source_language}-{target_language}",
    }
    for key in ("id", "docid", "segment_id"):
        if key in example:
            extra[key] = example[key]

    reference = str(translation[target_language]).strip()
    return as_qa_row(
        str(translation[source_language]).strip(),
        reference,
        aliases=[],
        fewshot_answer=reference,
        **extra,
    )


def parse_response(response: Any) -> str:
    text = str(response).strip()
    if not text:
        return ""

    if "```" in text:
        text = text.replace("```text", "```").replace("```translation", "```")
        parts = [part.strip() for part in text.split("```") if part.strip()]
        if parts:
            text = parts[0]

    labelled_matches = list(re.finditer(r"(?:translation|english|answer|final answer)\s*[:\-]\s*", text, re.I))
    if labelled_matches:
        text = text[labelled_matches[-1].end():].strip()

    echoed_prompt = _ECHOED_PROMPT_RE.search(text)
    if echoed_prompt:
        text = text[:echoed_prompt.start()].strip()

    text = _ANSWER_PREFIX_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return _normalize_space(text)


def extract_short_answer(response: Any) -> str:
    return parse_response(response)


def normalize_answer(answer: Any) -> str:
    return _normalize_space(parse_response(answer)).lower()


def cluster_key(response: Any, sample: QASample | None = None) -> str:
    del sample
    parsed = normalize_answer(response)
    return normalize_key(parsed) if parsed else normalize_key(response)


def _normalize_space(text: Any) -> str:
    return " ".join(str(text).strip().split())


def make_translation_adapter(cfg: dict[str, str]) -> Any:
    from types import SimpleNamespace
    from src.datasets._templates import build_judge_prompt_from_template, few_shot_prefix_from_template, format_prompt_from_template
    from src.labeling.judge import parse_judge_score

    ds_name = cfg["dataset_name"]
    src_lang = cfg["source_language"]
    tgt_lang = cfg["target_language"]

    def _extract_qa(example):
        return extract_translation_pair(example, source_language=src_lang, target_language=tgt_lang)

    def _format_prompt(question, prefix=None, choices=None):
        return format_prompt_from_template(question, ds_name, prefix=prefix)

    def _few_shot_prefix(few_shot_examples=None):
        return few_shot_prefix_from_template(few_shot_examples, ds_name)

    def _build_judge_prompt(record):
        return build_judge_prompt_from_template(record, ds_name)

    def _parse_judge_output(text):
        return parse_judge_score(text)

    return SimpleNamespace(
        extract_question_and_answer=_extract_qa,
        format_prompt=_format_prompt,
        few_shot_prefix=_few_shot_prefix,
        parse_response=parse_response,
        extract_short_answer=extract_short_answer,
        normalize_answer=normalize_answer,
        cluster_key=cluster_key,
        build_judge_prompt=_build_judge_prompt,
        parse_judge_output=_parse_judge_output,
    )
