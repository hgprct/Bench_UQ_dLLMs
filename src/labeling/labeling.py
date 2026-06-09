"""Correctness labeling orchestrator.

Selects greedy records, delegates to the appropriate labeling method
(exact match or LLM judge), and writes labels back.
"""

from __future__ import annotations

from typing import Any

from src.labeling.judge import judge_label_records_batch, response_text


def label_records(
    records: list[dict[str, Any]],
    dataset_module: Any,
    *,
    judge_model: str = "meta-llama/Llama-3.3-70B-Instruct",
    judge_tp: int | None = None,
    evaluator: Any = None,
    force: bool = False,
) -> int:
    """Label greedy records in-place. Returns count labeled."""
    to_label = _select_records_to_label(records)
    if not force:
        to_label = [r for r in to_label if r.get("final", {}).get("is_correct") is None]
    if not to_label:
        return 0

    method = getattr(dataset_module, "DEFAULT_LABEL_METHOD", "llm_judge")

    if method == "exact_match":
        labels, retried = _exact_match_batch(to_label, dataset_module)
    else:
        needs_gpu = getattr(dataset_module, "LABEL_GPU_MODE", "gpu") == "gpu"
        if needs_gpu and evaluator is None:
            from src.judge.evaluator import EvaluatorLLMLocal
            evaluator = EvaluatorLLMLocal(model_name=judge_model, tensor_parallel_size=judge_tp)
        labels, retried = judge_label_records_batch(
            to_label, evaluator,
            dataset_module.build_judge_prompt,
            dataset_module.parse_judge_output,
        )

    for i, (record, is_correct) in enumerate(zip(to_label, labels)):
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
        if r.get("final", {}).get("is_correct") is None:
            bad.add(pid)
    return bad


def _select_records_to_label(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select greedy records for labeling (exclude sampled)."""
    return [r for r in records if r.get("generation_mode") != "sampled"]


def _exact_match_batch(
    records: list[dict[str, Any]],
    dataset_module: Any,
) -> tuple[list[bool | None], set[int]]:
    """Label records using the dataset module's exact match logic."""
    labels = []
    for record in records:
        answer = response_text(record)
        parsed = dataset_module.normalize_answer(answer)
        ref = record.get("reference_answer", "")
        all_refs = record.get("aliases", [ref]) if record.get("aliases") else [ref]
        if hasattr(dataset_module, "exact_match_correct"):
            labels.append(dataset_module.exact_match_correct(parsed, all_refs))
        else:
            norm_refs = [dataset_module.normalize_answer(str(r)) for r in all_refs]
            labels.append(any(parsed == r for r in norm_refs) if parsed else False)
    return labels, set()
