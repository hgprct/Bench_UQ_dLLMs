"""HotpotQA multi-hop in-context QA dataset adapter."""

from __future__ import annotations

import re
import string
from typing import Any, Sequence

from src.datasets.qa_pair import QASample, as_qa_row
from src.utils.text import normalize_key, normalize_text
from src.labeling.judge import parse_judge_score

HF_NAME = "hotpot_qa"
DEFAULT_CONFIG_NAME = "distractor"
DEFAULT_SPLIT = "validation"
DEFAULT_LABEL_METHOD = "llm_judge"
LABEL_GPU_MODE = "gpu"
FEWSHOT_K = 0

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation + "‘’´`")
_SHORT_ANSWER_PATTERNS = (
    re.compile(r"^the answer is\s+(.+)$", flags=re.IGNORECASE),
    re.compile(r"^answer:\s*(.+)$", flags=re.IGNORECASE),
    re.compile(r"^final answer:\s*(.+)$", flags=re.IGNORECASE),
)


def _build_context_block(example: dict[str, Any]) -> str:
    titles = example.get("context", {}).get("title", [])
    sentences_list = example.get("context", {}).get("sentences", [])
    if not titles or not sentences_list:
        return ""
    paragraphs = []
    for title, sentences in zip(titles, sentences_list):
        text = "".join(sentences).strip()
        if text:
            paragraphs.append(f"[{title}]\n{text}")
    return "\n\n".join(paragraphs)


def extract_question_and_answer(example: dict[str, Any]) -> QASample:
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    context_block = _build_context_block(example)

    extra: dict[str, Any] = {
        "task_type": "multi_hop_qa",
        "context": context_block,
        "level": example.get("level", ""),
        "type": example.get("type", ""),
    }
    doc_id = example.get("id")
    if doc_id is not None:
        extra["id"] = doc_id

    supporting = example.get("supporting_facts")
    if supporting:
        extra["supporting_facts"] = supporting

    return as_qa_row(
        question,
        answer,
        aliases=[answer],
        fewshot_answer=answer,
        **extra,
    )


def format_prompt(
    question: str,
    prefix: str | None = None,
    choices: Sequence[dict[str, str]] | None = None,
    *,
    context: str | None = None,
) -> str:
    parts = [
        "Answer the following question based on the provided supporting documents. "
        "Read all documents carefully and combine information from multiple "
        "documents if needed.",
    ]

    if context:
        parts.append(f"Supporting documents:\n\n{context}")

    if prefix:
        parts.append(str(prefix).strip())

    parts.append(f"Question: {str(question).strip()}\nAnswer:")

    return "\n\n".join(parts).strip()


def few_shot_prefix(few_shot_examples: list[dict[str, Any]] | None = None) -> str:
    if not few_shot_examples:
        return ""
    examples = []
    for ex in few_shot_examples:
        q = str(ex.get("question", "")).strip()
        a = str(ex.get("answer", "")).strip()
        examples.append(f"Question: {q}\nAnswer: {a}")
    return "Here are some examples:\n\n" + "\n\n".join(examples)


def normalize_answer(text: Any) -> str:
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
    lines = text.strip().splitlines()
    if lines:
        last = lines[-1].strip()
        if last:
            return last
    return text


def parse_response(response: Any) -> str:
    raw = str(response).strip()
    for pattern in _SHORT_ANSWER_PATTERNS:
        match = pattern.match(raw)
        if match:
            return match.group(1).strip()
    return raw


def cluster_key(response: Any, sample: QASample | None = None) -> str:
    del sample
    return normalize_key(normalize_answer(parse_response(response)))


def _f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_correct(prediction: str, gold_answers: Sequence[Any]) -> bool:
    pred_norm = normalize_answer(prediction)
    return any(pred_norm == normalize_answer(str(g)) for g in gold_answers)


def build_judge_prompt(record: dict[str, Any]) -> str:
    question = record.get("question", "")
    context = record.get("context", "")

    ref = record.get("reference_answer", "")
    aliases = record.get("aliases", [ref]) if record.get("aliases") else [ref]
    refs = ", ".join(str(a) for a in aliases if a)

    final = record.get("final", {})
    candidate = str(final.get("answer", final.get("response", "")) or "")

    parts = [
        "Task: You evaluate whether the model response correctly answers a "
        "multi-hop question. The question requires combining information from "
        "the provided context documents.",
    ]

    if context:
        parts.append(f"Context:\n{context}")

    parts.append(f"Question: {question}")
    parts.append(f"Acceptable answers: {refs}")
    parts.append(f"Response: {candidate}")
    parts.append(
        "Accept conversational hedges if the core answer matches any reference. "
        "Answer only 0 (incorrect or uncommitted) or 1 (correct). "
        "Output a single digit, nothing else.\nScore:"
    )

    return "\n\n".join(parts)


def parse_judge_output(text: Any) -> bool | None:
    return parse_judge_score(text)
