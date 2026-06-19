"""LiveBench reasoning subset adapter.

HF dataset: livebench/reasoning (test split). Loaded from the local parquet
(``livebench/test-00000-of-00001.parquet``) per DATASET_CONFIGS so generation runs
offline.

Row schema:
  question_id            : str
  category               : str   ("reasoning")
  task                   : str   e.g. "zebra_puzzle", "spatial", "web_of_lies_v2"
  turns                  : list[str]  the prompt (single self-contained turn)
  ground_truth           : str   the reference answer
  level                  : int
  livebench_release_date : timestamp
  livebench_removal_date : str

LiveBench prompts are self-contained (they already include the full instruction
and the required answer format), so ``format_prompt`` passes the turn text through
and only prepends an optional few-shot prefix. Judged against ``ground_truth``.
"""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score


def _prompt_text(turns: Any) -> str:
    if isinstance(turns, (list, tuple)):
        return "\n\n".join(str(t).strip() for t in turns if str(t).strip())
    return str(turns or "").strip()


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract prompt + ground-truth from a LiveBench reasoning row."""
    question = _prompt_text(example.get("turns"))
    answer = str(example.get("ground_truth", "")).strip()
    if not question or not answer:
        return None
    return QASample(
        question=question,
        reference_answer=answer,
        subject=str(example.get("task", "")).strip(),
        id=str(example.get("question_id", "")),
    )


# Generation functions
def format_prompt(qa_sample: QASample, fewshot_prefix: str | None = None) -> str:
    """Format a LiveBench prompt (the turn is already a complete instruction)."""
    prefix = fewshot_prefix or ""
    return f"{prefix}{qa_sample['question']}"


def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for LiveBench reasoning."""
    if not few_shot_examples:
        return ""
    blocks = [
        f"{ex['question']}\nAnswer: {ex.get('reference_answer', '')}"
        for ex in few_shot_examples
    ]
    return "Here are examples:\n\n" + "\n\n".join(blocks) + "\n\n"


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of a LiveBench reasoning record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key"
    prompt = "Task : You evaluate whether the model response reaches the correct answer.\n"
    prompt += "The task instructions and required answer format are in the question.\n"
    prompt += f"Question:\n{record['question']}\n\n"
    prompt += f"Correct answer: {record.get('reference_answer', '')}\n"
    prompt += f"Model response: {record['raw_answer']}\n"
    prompt += "Answer 1 if the model's final answer matches the correct answer, else 0.\n"
    prompt += "Output a single digit, nothing else.\nScore:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a LiveBench record into a label."""
    return parse_judge_score(text)
