#!/usr/bin/env python3
"""Join the OK-VQA questions + annotations into a single JSONL.

OK-VQA ships two upstream files for the COCO val2014 split:

  * ``OpenEnded_mscoco_val2014_questions.json`` -- {"questions": [{image_id,
    question, question_id}, ...]} (no answers).
  * ``mscoco_val2014_annotations.json``         -- {"annotations": [{question_id,
    image_id, answers: [{answer, raw_answer, ...} x10], ...}, ...]}.

The ground-truth answer for each question is the majority of the ten normalised
human ``answer`` strings (OK-VQA's canonical single answer). This script joins
the two by ``question_id`` and writes one clean record per line, resolving the
COCO image to a repo-relative path::

    {"id", "image_id", "image_path", "question", "answer", "answers"}

``answers`` keeps the distinct human answers (the judge may accept any). The
result (``datasets/OKVQA/test.jsonl``) is what the ``okvqa`` adapter consumes via
``DATASET_CONFIGS[Dataset.okvqa].local_path``.

Usage:
    python scripts/prepare_okvqa.py            # questions + annotations -> test.jsonl
    python scripts/prepare_okvqa.py QUESTIONS ANNOTATIONS DST
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_DIR = _REPO_ROOT / "datasets" / "OKVQA"
_DEFAULT_QUESTIONS = _DATASET_DIR / "OpenEnded_mscoco_val2014_questions.json"
_DEFAULT_ANNOTATIONS = _DATASET_DIR / "mscoco_val2014_annotations.json"
_DEFAULT_DST = _DATASET_DIR / "test.jsonl"


def _coco_image_path(image_id: int) -> str:
    """Repo-relative path to a COCO val2014 image (e.g. id 42 -> ...000000000042.jpg)."""
    fname = f"COCO_val2014_{int(image_id):012d}.jpg"
    return (Path("datasets/OKVQA/val2014") / fname).as_posix()


def _load_answers(annotations_path: Path) -> dict[int, dict]:
    """Map question_id -> {"answer": majority, "answers": [distinct]} from annotations."""
    if not annotations_path.is_file():
        print(
            f"[prepare_okvqa] annotations not found: {annotations_path}. "
            "Ground-truth answers will be empty; add the OK-VQA annotations file "
            "and re-run to populate them."
        )
        return {}
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for ann in data.get("annotations", []):
        answers = [str(a.get("answer", "")).strip() for a in ann.get("answers", [])]
        answers = [a for a in answers if a]
        if not answers:
            continue
        majority = Counter(answers).most_common(1)[0][0]
        out[int(ann["question_id"])] = {
            "answer": majority,
            "answers": sorted(set(answers)),
        }
    return out


def convert(questions_path: Path, annotations_path: Path, dst: Path) -> int:
    questions = json.loads(questions_path.read_text(encoding="utf-8")).get("questions", [])
    answers_by_qid = _load_answers(annotations_path)

    records = []
    for q in questions:
        question = str(q.get("question", "")).strip()
        image_id = q.get("image_id")
        question_id = q.get("question_id")
        if not question or image_id is None or question_id is None:
            continue
        gt = answers_by_qid.get(int(question_id), {})
        records.append(
            {
                "id": str(question_id),
                "image_id": str(image_id),
                "image_path": _coco_image_path(image_id),
                "question": question,
                "answer": gt.get("answer", ""),
                "answers": gt.get("answers", []),
            }
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    return len(records)


def main(argv: list[str]) -> None:
    questions = Path(argv[1]) if len(argv) > 1 else _DEFAULT_QUESTIONS
    annotations = Path(argv[2]) if len(argv) > 2 else _DEFAULT_ANNOTATIONS
    dst = Path(argv[3]) if len(argv) > 3 else _DEFAULT_DST
    n = convert(questions, annotations, dst)
    print(f"Wrote {n} OK-VQA questions to {dst}")


if __name__ == "__main__":
    main(sys.argv)
