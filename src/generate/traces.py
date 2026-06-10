"""Output serialization: answers JSONL + top-k logprobs NPZ."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def save_results(
    output_dir: str,
    run_config: dict[str, Any],
    qa_pairs: list[dict],
    prompts: list[str],
    answers: list[str],
    topk_data: dict[str, Any],
    dataset_key: str,
    *,
    parse_answer: Any = None,
) -> dict[str, Any]:
    """Write answers.jsonl, topk_logprobs.npz, and config.json."""
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.json")
    answers_path = os.path.join(output_dir, "answers.jsonl")
    topk_path = os.path.join(output_dir, "topk_logprobs.npz")

    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=True)

    records = _build_answer_records(
        qa_pairs, prompts, answers, dataset_key, parse_answer=parse_answer,
    )
    with open(answers_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, allow_nan=False) + "\n")

    arrays = {}
    for key, val in topk_data.items():
        if hasattr(val, "numpy"):
            arrays[key] = val.detach().cpu().numpy()
        else:
            arrays[key] = np.asarray(val)
    np.savez_compressed(topk_path, **arrays)

    metadata = {
        "run_id": run_config.get("run_id", "unknown"),
        "num_samples": len(records),
        "run_config": run_config,
        "files": {
            "config": "config.json",
            "answers": "answers.jsonl",
            "topk_logprobs": "topk_logprobs.npz",
        },
        "topk_arrays": sorted(arrays),
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "config_path": config_path,
        "answers_path": answers_path,
        "topk_path": topk_path,
        "num_examples": len(records),
    }


def _build_answer_records(
    qa_pairs: list[dict],
    prompts: list[str],
    answers: list[str],
    dataset_key: str,
    *,
    parse_answer: Any = None,
) -> list[dict]:
    records = []
    for i, (qa, prompt, answer) in enumerate(zip(qa_pairs, prompts, answers)):
        parsed = parse_answer(answer) if parse_answer else None
        record = dict(qa) if isinstance(qa, dict) else {}
        if not record.get("aliases"):
            record["aliases"] = [record.get("reference_answer", "")]
        record.update({
            "example_id": _example_id(qa, i),
            "sample_id": i,
            "dataset": dataset_key,
            "prompt": prompt,
            "raw_answer": answer,
            "parsed_answer": parsed,
        })
        records.append(record)
    return records


def _example_id(qa_item: Any, sample_id: int) -> str:
    if isinstance(qa_item, dict):
        for key in ("sample_example_id", "id", "example_id", "question_id"):
            if key in qa_item and str(qa_item[key]).strip():
                return str(qa_item[key])
    return str(sample_id)
