"""CHAMP (Concept and Hint-Annotated Math Problems) dataset adapter.

Source: https://github.com/YujunMao1/CHAMP -- 270 challenging competition-style
maths problems across five categories (Combinatorics, Inequality, Number-Theory,
Polynomial, Sequence). Loaded from the slim local JSONL produced by
``scripts/prepare_champ.py`` (``CHAMP/champ.jsonl``); each record is:

  identifier : str   e.g. "P_Combinatorics_1"
  question   : str   the problem statement
  answer     : str   the reference final answer
  category   : str   one of the five categories
  solution   : str   the worked solution (joined steps)

Unlike MATH/competition_math, CHAMP answers are frequently symbolic or given in
multiple equivalent forms (e.g. "C(10, 5), or equivalently 252", "2^(n-1)",
"n(n-3)/2", "No"), so exact-match is unreliable.
"""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract problem + boxed reference answer from a CHAMP record."""
    question = str(example.get("question", "")).strip()
    ground_truth_answer = str(example.get("solution", "")).strip()

    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="champ",
        split="train"
    )


# Generation functions
def format_prompt(question: str) -> str:
    """Format a CHAMP prompt, asking for a final ``\\boxed{}`` answer."""
    instruction = (
        "Please briefly solve the following maths problem:"
    )
    return f"{instruction} {question}\nOnly return the main maths steps and the final answer, do not include any verbal additional commentary."