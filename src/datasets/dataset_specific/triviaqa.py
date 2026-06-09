"""TriviaQA dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample, as_qa_row
from src.utils.text import normalize_aliases, normalize_key, parse_text_response
from src.labeling.judge import parse_judge_score

HF_NAME = "mandarjoshi/trivia_qa"
DEFAULT_CONFIG_NAME = "rc"
DEFAULT_SPLIT = "test"
DEFAULT_LABEL_METHOD = "llm_judge"
LABEL_GPU_MODE = "gpu"
FEWSHOT_K = 0


def extract_question_and_answer(example: dict[str, Any]) -> QASample:
    answer = example.get("answer", {})
    aliases = answer.get("aliases", []) if isinstance(answer, dict) else []
    reference_answer = answer.get("value", "") if isinstance(answer, dict) else answer
    return as_qa_row(example.get("question", ""), reference_answer, normalize_aliases(aliases))


def format_prompt(
    question: str,
    additional_elements: dict[str, Any] | None = None,
) -> str:
    return "Question: " + question + "\nAnswer:"


def few_shot_prefix(few_shot_examples: list[dict[str, Any]] | None = None) -> str:
    return None


def parse_response(response: Any) -> str:
    return parse_text_response(response)


def normalize_answer(answer: Any) -> str:
    return parse_text_response(answer).lower()


def cluster_key(response: Any, sample: QASample | None = None) -> str:
    del sample
    parsed = normalize_answer(response)
    return normalize_key(parsed) if parsed else normalize_key(response)


def build_judge_prompt(record: dict[str, Any]) -> str:
    assert "question" in record, "Record must contain 'question' key"
    assert "reference_answer" in record, "Record must contain 'reference_answer' key"
    assert "final" in record, "Record must contain 'final' key with model response"

    ref = record.get("reference_answer", "")
    aliases = record.get("answer_aliases", record.get("aliases"))
    gold_answers = aliases if isinstance(aliases, list) and aliases else [ref]
    refs = ", ".join(a for a in gold_answers if a)

    final = record.get("final", {})
    candidate = str(final.get("answer", final.get("response", "")) or "")

    prompt = "Task : You evaluate whether the model response correctly answers the question.\n"
    prompt += "Accept conversational hedges if the core answer matches any reference alias.\n"
    prompt += f"Question: {record['question']}\n"
    prompt += f"Acceptable response: {refs}\n"
    prompt += f"Model response: {candidate}\n"
    prompt += "Answer only 0 (incorrect or uncommitted) or 1 (correct). Output a single digit, nothing else."
    prompt += "Score:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    return parse_judge_score(text)
