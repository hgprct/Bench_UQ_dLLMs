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
and the required answer format), so ``format_prompt`` passes the turn text through.
"""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample

# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract prompt + ground-truth from a LiveBench reasoning row."""
    question = str(example.get("turns", "")[0]).strip()
    ground_truth_answer = str(example.get("ground_truth", "")).strip()
    qid = str(example.get("question_id", "")).strip() or id

    return QASample(
        id=qid,
        image_id=qid,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="livebench_reasoning",
        split="test"
    )


# Generation functions
def format_prompt(question: str) -> str:
    """LiveBench prompts are self-contained; pass the turn text through."""
    return question
