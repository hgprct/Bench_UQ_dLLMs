#!/usr/bin/env python3
"""Flatten the raw DrivingVQA dump (``datasets/DrivingVQA/test.json``) into a JSONL.

The upstream file is a single JSON object keyed by sample id; each value is a
driving-theory multiple-choice question over a dashcam image::

    {
      "img_filename": "images/0000.jpg",
      "questions": "I could take"            # str, or list[str] for 2 sub-questions
      "possible_answers": {"A": "...", "B": "...", ...},
      "true_answers": ["B", "C", "D"],        # one or more correct option letters
      "explanation": "...",
      ...
    }

This script flattens it into one clean record per line, keeping only what the
``drivingvqa`` adapter needs and resolving the image to a repo-relative path::

    {"id", "image_path", "questions", "possible_answers", "true_answers", "explanation"}

The result (``datasets/DrivingVQA/test.jsonl``) is what the adapter consumes via
``DATASET_CONFIGS[Dataset.drivingvqa].local_path``.

Usage:
    python scripts/prepare_drivingvqa.py            # test.json -> test.jsonl
    python scripts/prepare_drivingvqa.py SRC DST
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_DIR = _REPO_ROOT / "datasets" / "DrivingVQA"
_DEFAULT_SRC = _DATASET_DIR / "test.json"
_DEFAULT_DST = _DATASET_DIR / "test.jsonl"


def convert(src: Path, dst: Path) -> int:
    data = json.loads(src.read_text(encoding="utf-8"))
    records = []
    for key, row in data.items():
        img_filename = str(row.get("img_filename", "")).strip()
        possible_answers = row.get("possible_answers") or {}
        true_answers = row.get("true_answers") or []
        if not img_filename or not possible_answers or not true_answers:
            continue
        # img_filename is relative to the dataset dir (e.g. "images/0000.jpg");
        # store a repo-relative path so the adapter can open it directly.
        image_path = (Path("datasets/DrivingVQA") / img_filename).as_posix()
        records.append(
            {
                "id": str(row.get("id", key)),
                "image_path": image_path,
                "questions": row.get("questions"),
                "possible_answers": possible_answers,
                "true_answers": list(true_answers),
                "explanation": str(row.get("explanation", "")).strip(),
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
    print(f"Wrote {n} DrivingVQA questions to {dst}")


if __name__ == "__main__":
    main(sys.argv)
