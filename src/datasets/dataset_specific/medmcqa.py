"""MedMCQA medical multiple-choice dataset adapter.

HF dataset: openlifescienceai/medmcqa (train / validation / test).

Row schema:
  id           : str
  question     : str
  opa..opd     : str          the four answer options
  cop          : ClassLabel   correct option, names ["a","b","c","d"]
                              (loaded as int 0-3; raw json may give "a"-"d")
  choice_type  : str
  exp          : str          rationale / explanation
  subject_name : str
  topic_name   : str

Four-way MC. The correct option is rendered as both a letter (A-D) and its text,
and judged against the model response (CoT answers are verbose, so an LLM judge
is more robust than letter exact-match).
"""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score

_LETTERS = ("A", "B", "C", "D")


def _correct_index(cop: Any) -> int | None:
    """Map ``cop`` (int 0-3 or letter 'a'-'d') to a 0-based index."""
    if isinstance(cop, bool):
        return None
    if isinstance(cop, int):
        return cop if 0 <= cop <= 3 else None
    s = str(cop).strip().lower()
    if s in ("a", "b", "c", "d"):
        return ord(s) - ord("a")
    if s.isdigit() and 0 <= int(s) <= 3:
        return int(s)
    return None


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract question + options + correct answer from a MedMCQA row."""
    question = str(example.get("question", "")).strip()
    options = [str(example.get(k, "")).strip() for k in ("opa", "opb", "opc", "opd")]
    idx = _correct_index(example.get("cop"))
    if not question or idx is None or not any(options):
        return None
    correct_letter = _LETTERS[idx]
    correct_text = options[idx]
    return QASample(
        question=question,
        reference_answer=f"{correct_letter}. {correct_text}",
        options=options,
        subject=str(example.get("subject_name", "")).strip(),
        id=str(example.get("id", "")),
    )


def _options_block(options: list[str]) -> str:
    return "\n".join(f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(options))


# Generation functions
def format_prompt(qa_sample: QASample, fewshot_prefix: str | None = None) -> str:
    """Format a MedMCQA prompt with lettered options."""
    instruction = (
        "Answer the following multiple-choice medical question. Reason briefly, "
        "then state the correct option as a single letter (A, B, C, or D)."
    )
    options = qa_sample.get("options") or []
    prefix = fewshot_prefix or ""
    return (
        f"{instruction}\n\n{prefix}Question: {qa_sample['question']}\n"
        f"{_options_block(options)}\n\nAnswer:"
    )


def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for MedMCQA."""
    if not few_shot_examples:
        return ""
    blocks = []
    for ex in few_shot_examples:
        opts = _options_block(ex.get("options") or [])
        blocks.append(
            f"Question: {ex['question']}\n{opts}\nAnswer: {ex.get('reference_answer', '')}"
        )
    return "Here are examples:\n\n" + "\n\n".join(blocks) + "\n\n"


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of a MedMCQA record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key"
    options = record.get("options") or []
    prompt = "Task : You evaluate whether the model picked the correct option.\n"
    prompt += f"Question: {record['question']}\n"
    if options:
        prompt += f"Options:\n{_options_block(options)}\n"
    prompt += f"Correct option: {record.get('reference_answer', '')}\n"
    prompt += f"Model response: {record['raw_answer']}\n"
    prompt += "Answer 1 if the model's chosen option matches the correct option, else 0.\n"
    prompt += "Output a single digit, nothing else.\nScore:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a MedMCQA record into a label."""
    return parse_judge_score(text)
