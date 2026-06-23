#!/usr/bin/env python3
"""Convert the raw CHAMP dump (``CHAMP/v0.json``) into a slim JSONL.

The upstream file (https://github.com/YujunMao1/CHAMP) is a single ~72 MB JSON
object ``{problems, concepts, hints}`` where ``problems`` is a dict keyed by
identifier; each problem also carries the full transcripts of every benchmarked
model under ``conversations`` (the bulk of the size). The pipeline only needs the
problem statement, the reference answer, the category, and the worked solution,
so this script flattens ``problems`` into one clean record per line:

    {"identifier", "question", "answer", "category", "solution"}

``question``/``answer`` use the cleaned ``_text``/``_answer`` fields (the raw
variants wrap maths in ``@@...@@`` delimiters); ``solution`` joins the cleaned
solution-step texts. The result (``CHAMP/champ.jsonl``) is what the ``champ``
dataset adapter consumes via ``DATASET_CONFIGS[Dataset.champ].local_path``.

Usage:
    python scripts/prepare_champ.py            # CHAMP/v0.json -> CHAMP/champ.jsonl
    python scripts/prepare_champ.py SRC DST
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _REPO_ROOT / "CHAMP" / "v0.json"
_DEFAULT_DST = _REPO_ROOT / "CHAMP" / "champ.jsonl"


def _solution_text(problem: dict) -> str:
    """Join the cleaned solution-step texts into a single worked solution."""
    steps = (problem.get("solution") or {}).get("steps") or []
    parts = [str(step.get("_text", "")).strip() for step in steps]
    return "\n".join(p for p in parts if p)


def convert(src: Path, dst: Path) -> int:
    data = json.loads(src.read_text(encoding="utf-8"))
    problems = data["problems"]
    records = []
    for identifier, problem in problems.items():
        question = str(problem.get("_text", "")).strip()
        answer = str(problem.get("_answer", "")).strip()
        if not question or not answer:
            continue
        records.append(
            {
                "identifier": str(problem.get("identifier", identifier)),
                "question": question,
                "answer": answer,
                "category": str(problem.get("category", "")).strip(),
                "solution": _solution_text(problem),
            }
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    return len(records)


def main(argv: list[str]) -> None:
    src = Path(argv[1]) if len(argv) > 1 else _DEFAULT_SRC
    dst = Path(argv[2]) if len(argv) > 2 else _DEFAULT_DST
    n = convert(src, dst)
    print(f"Wrote {n} CHAMP problems to {dst}")


if __name__ == "__main__":
    main(sys.argv)
