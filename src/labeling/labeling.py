"""Correctness labeling orchestrator.

Public interface
----------------
label_qa_samples(samples, dataset_key, ...)
    Clean entry point: given a list of QASample objects with raw_answer already set,
    writes the label field on each sample in-place.

label_records(records, dataset_key, ...)
    Pipeline entry point: given full example records from examples.jsonl,
    labels greedy records in-place and returns the count labeled.
"""

from __future__ import annotations

from typing import Any

from src.config import DATASET_CONFIGS, Dataset
from src.datasets.qa_pair import QASample
from src.labeling.judge import judge_label_records_batch, response_text


# ---------------------------------------------------------------------------
# Clean QASample-level interface
# ---------------------------------------------------------------------------

def label_qa_samples(
    samples: list[QASample],
    dataset_key: str,
    *,
    evaluator: Any = None,
    judge_model: str = "meta-llama/Llama-3.3-70B-Instruct",
    judge_tp: int | None = None,
) -> None:
    """Label QASample objects in-place.

    Reads raw_answer from each sample and writes its label field (True/False/None).
    All samples are labeled regardless of generation_mode.
    """
    from src.registry import get_dataset_module
    ds_config = DATASET_CONFIGS[Dataset(dataset_key)]
    dataset_module = get_dataset_module(dataset_key)

    if ds_config.label_method == "exact_match":
        labels, _ = _exact_match_batch(samples, dataset_module)
    else:
        if ds_config.label_gpu_mode == "gpu" and evaluator is None:
            from src.judge.evaluator import EvaluatorLLMLocal
            evaluator = EvaluatorLLMLocal(model_name=judge_model, tensor_parallel_size=judge_tp)
        labels, _ = judge_label_records_batch(
            samples, evaluator,
            dataset_module.build_judge_prompt,
            dataset_module.parse_judge_output,
        )

    for sample, is_correct in zip(samples, labels):
        sample["label"] = is_correct


# ---------------------------------------------------------------------------
# Pipeline record-level interface (works on full examples.jsonl records)
# ---------------------------------------------------------------------------

def label_records(
    records: list[dict[str, Any]],
    dataset_key: str,
    *,
    judge_model: str = "meta-llama/Llama-3.3-70B-Instruct",
    judge_tp: int | None = None,
    evaluator: Any = None,
    force: bool = False,
) -> int:
    """Label greedy example records in-place. Returns count labeled."""
    to_label = _select_records_to_label(records)
    if not force:
        to_label = [r for r in to_label if r.get("label") is None]
    if not to_label:
        return 0

    from src.registry import get_dataset_module
    ds_config = DATASET_CONFIGS[Dataset(dataset_key)]
    dataset_module = get_dataset_module(dataset_key)

    if ds_config.label_method == "exact_match":
        labels, retried = _exact_match_batch(to_label, dataset_module)
    else:
        if ds_config.label_gpu_mode == "gpu" and evaluator is None:
            from src.judge.evaluator import EvaluatorLLMLocal
            evaluator = EvaluatorLLMLocal(model_name=judge_model, tensor_parallel_size=judge_tp)
        labels, retried = judge_label_records_batch(
            to_label, evaluator,
            dataset_module.build_judge_prompt,
            dataset_module.parse_judge_output,
        )

    method = ds_config.label_method
    for i, (record, is_correct) in enumerate(zip(to_label, labels)):
        record["label"] = is_correct
        final = record.setdefault("final", {})
        final["is_correct"] = is_correct
        if is_correct is None:
            final["correctness_method"] = f"{method}_ambiguous"
        elif i in retried:
            final["correctness_method"] = f"{method}_retry"
        else:
            final["correctness_method"] = method

    return len(to_label)


def unlabeled_prompt_ids(records: list[dict[str, Any]]) -> set[str]:
    """Return prompt IDs where the greedy record has no valid label."""
    selected = _select_records_to_label(records)
    bad: set[str] = set()
    for r in selected:
        pid = str(r.get("qa_example_id", ""))
        if r.get("label") is None:
            bad.add(pid)
    return bad


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_records_to_label(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select greedy records for labeling (exclude sampled)."""
    return [r for r in records if r.get("generation_mode") != "sampled"]


def _exact_match_batch(
    records: list[Any],
    dataset_module: Any,
) -> tuple[list[bool | None], set[int]]:
    """Label records using the dataset module's exact match logic."""
    labels: list[bool | None] = []
    for record in records:
        raw_answer = response_text(record)
        if raw_answer == "":
            labels.append(False)
            continue
        parsed = dataset_module.parse_answer(raw_answer)
        record["parsed_answer"] = parsed
        ref = record.get("reference_answer", "")
        all_refs = record.get("aliases") or [ref]
        labels.append(any(parsed == r for r in all_refs) if parsed else False)
    return labels, set()
