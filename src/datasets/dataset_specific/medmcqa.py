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

Four-way MC. The correct option is rendered as both a letter (A-D) and its text.
"""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample

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
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract question + options + correct answer from a MedMCQA row."""
    question = str(example.get("question", "")).strip()
    options = [str(example.get(k, "")).strip() for k in ("opa", "opb", "opc", "opd")]
    ground_truth_answer = options[_correct_index(example.get("cop"))] if _correct_index(example.get("cop")) is not None else None

    
    
    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question, options),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="medmcqa",
        split="validation"
    )


# Generation functions
def format_prompt(question: str, options: list[str]) -> str:
    """Format a MedMCQA prompt with lettered options."""
    question += "\n-".join(f"{opt}" for opt in options)
    return (
        f"{question}\n"
        "Please first write down the answer and provide a brief explanation for your answer.\n"
    )