"""Competition mathematics (MATH) dataset adapter.

HF dataset: qwedsacf/competition_math (single ``train`` split, 12.5k rows).

Row schema:
  problem   : str   the problem statement
  level     : str   e.g. "Level 5"
  type      : str   e.g. "Algebra"
  solution  : str   worked solution ending in a final ``\\boxed{...}``

The reference answer is the content of the solution's final ``\\boxed{}``; the
model is prompted to put its own answer in ``\\boxed{}``.
"""

from __future__ import annotations
from typing import Any

from src.datasets.qa_pair import QASample

# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract problem + boxed reference answer from a MATH row."""
    question = str(example.get("problem", "")).strip()
    ground_truth_answer = str(example.get("solution", "")).strip()

    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="competition_math",
        split="train"
    )


# Generation functions
def format_prompt(question: str) -> str:
    """Format a MATH prompt, asking for a final ``\\boxed{}`` answer."""
    instruction = (
        "Please briefly solve the following maths problem:"
    )
    return f"{instruction} {question}\nOnly return the main maths steps and the final answer, do not include any verbal additional commentary."