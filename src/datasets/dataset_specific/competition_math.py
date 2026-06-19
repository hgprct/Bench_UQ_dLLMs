"""Competition mathematics (MATH) dataset adapter.

HF dataset: qwedsacf/competition_math (single ``train`` split, 12.5k rows).

Row schema:
  problem   : str   the problem statement
  level     : str   e.g. "Level 5"
  type      : str   e.g. "Algebra"
  solution  : str   worked solution ending in a final ``\\boxed{...}``

The reference answer is the content of the solution's final ``\\boxed{}``; the
model is prompted to put its own answer in ``\\boxed{}`` and is scored by
exact-match after boxed extraction + light LaTeX normalisation.
"""

from __future__ import annotations

import re
from typing import Any

from src.datasets.qa_pair import QASample

_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _extract_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` (brace-balanced)."""
    matches = list(_BOXED_RE.finditer(str(text)))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    out: list[str] = []
    for ch in str(text)[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def _normalize_math(s: str) -> str:
    """Light normalisation so equivalent boxed answers compare equal."""
    s = str(s).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" ", "")
    s = s.rstrip(".")
    return s


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract problem + boxed reference answer from a MATH row."""
    problem = str(example.get("problem", "")).strip()
    solution = str(example.get("solution", "")).strip()
    if not problem:
        return None
    boxed = _extract_boxed(solution)
    if boxed is None:
        return None
    return QASample(
        question=problem,
        reference_answer=_normalize_math(boxed),
        raw_reference_answer=solution,
        level=str(example.get("level", "")),
        subject=str(example.get("type", "")),
    )


# Generation functions
def format_prompt(qa_sample: QASample, fewshot_prefix: str | None = None) -> str:
    """Format a MATH prompt, asking for a final ``\\boxed{}`` answer."""
    instruction = (
        "Solve the following problem. Reason step by step, then give the final "
        "answer enclosed in \\boxed{}."
    )
    prefix = fewshot_prefix or ""
    return f"{instruction}\n\n{prefix}Problem: {qa_sample['question']}\n\nSolution:"


def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for MATH (problem + worked solution)."""
    if not few_shot_examples:
        return ""
    blocks = [
        f"Problem: {ex['question']}\n\nSolution: {ex.get('raw_reference_answer', '')}"
        for ex in few_shot_examples
    ]
    return "Here are worked examples:\n\n" + "\n\n".join(blocks) + "\n\n"


def parse_answer(raw_answer: str) -> str:
    """Extract + normalise the model's final boxed answer (fallback: last line)."""
    boxed = _extract_boxed(raw_answer)
    if boxed is not None:
        return _normalize_math(boxed)
    text = str(raw_answer).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    last_line = text.splitlines()[-1].strip() if text else ""
    return _normalize_math(last_line)
