"""TriviaQA dataset adapter.

Loaded from the local sampled CSV (``trivia_qa/sampled/test.csv``); each row is:

  id       : str   the TriviaQA question id (e.g. "wh_4198")
  question : str   the question text
  answer   : str   a Python-literal list of accepted answer aliases,
                   e.g. "['Ear (disambiguation)', 'EAR']"
"""

from __future__ import annotations

import ast
from typing import Any

from src.datasets.qa_pair import QASample


def _parse_aliases(raw: Any) -> list[str]:
    """Parse the CSV ``answer`` field (a Python-literal list) into raw alias strings."""
    if isinstance(raw, (list, tuple)):
        return [str(a) for a in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [raw]
    if isinstance(value, (list, tuple)):
        return [str(a) for a in value]
    return [str(value)]


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract question and answer aliases from a sampled TriviaQA CSV row."""
    question = str(example.get("question", "")).strip()
    ground_truth_answer = _parse_aliases(example.get("answer"))
    if not question or not ground_truth_answer:
        return None

    qid = str(example.get("id", "")).strip()
    return QASample(
        id=qid,
        image_id=qid,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="triviaqa",
        split="test"
    )


# Generation functions
def format_prompt(question: str) -> str:
    """Format a prompt for TriviaQA."""
    return "Question: " + question + "\nAnswer:"
