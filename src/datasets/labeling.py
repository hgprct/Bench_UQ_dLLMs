"""Correctness labeling for greedy answers."""

from __future__ import annotations

from typing import Any


def label_records(
    records: list[dict[str, Any]],
    dataset_module: Any,
    method: str,
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

    if method == "exact_match":
        labels = _label_records_exact_match(to_label, dataset_module)
        retried: set[int] = set()
    else:
        if evaluator is None:
            from src.judge.evaluator import EvaluatorLLMLocal
            evaluator = EvaluatorLLMLocal(model_name=judge_model, tensor_parallel_size=judge_tp)
        labels, retried = _label_records_llm_judge(to_label, dataset_module, evaluator)

    for i, (record, is_correct) in enumerate(zip(to_label, labels)):
        final = record.setdefault("final", {})
        final["is_correct"] = is_correct
        if is_correct is None:
            final["correctness_method"] = "llm_judge_ambiguous"
        elif i in retried:
            final["correctness_method"] = "llm_judge_retry"
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


# ---------------------------------------------------------------------------
# Exact match path (gsm8k)
# ---------------------------------------------------------------------------

def _label_records_exact_match(
    records: list[dict[str, Any]],
    dataset_module: Any,
) -> list[bool]:
    labels: list[bool] = []
    for record in records:
        final = record.get("final", {})
        answer = str(final.get("answer", final.get("response", "")))
        parsed = dataset_module.normalize_answer(answer)
        ref = record.get("reference_answer", "")
        all_refs = record.get("aliases", [ref]) if record.get("aliases") else [ref]
        labels.append(dataset_module.exact_match_correct(parsed, all_refs))
    return labels


# ---------------------------------------------------------------------------
# LLM judge path (all other datasets)
# ---------------------------------------------------------------------------

def _response_text(record: dict[str, Any]) -> str:
    final = record.get("final", {})
    return str(final.get("answer", final.get("response", "")) or "")


def _label_records_llm_judge(
    records: list[dict[str, Any]],
    dataset_module: Any,
    evaluator: Any,
) -> tuple[list[bool | None], set[int]]:
    """Returns (labels, retried_indices)."""
    if not hasattr(dataset_module, "build_judge_prompt"):
        raise ValueError(
            f"Dataset module '{dataset_module.__name__}' does not implement "
            "build_judge_prompt/parse_judge_output; use exact_match instead."
        )

    labels: list[bool | None] = [None] * len(records)
    judge_indices: list[int] = []
    for i, record in enumerate(records):
        if _response_text(record).strip() == "":
            labels[i] = False
        else:
            judge_indices.append(i)

    if not judge_indices:
        return labels, set()

    judge_records = [records[i] for i in judge_indices]
    prompts = [dataset_module.build_judge_prompt(r) for r in judge_records]

    raw_outputs = evaluator.predict_batch(prompts, temperature=0.0, max_tokens=16)
    for j, (idx, out) in enumerate(zip(judge_indices, raw_outputs)):
        labels[idx] = dataset_module.parse_judge_output(out)

    retry_indices = [idx for idx in judge_indices if labels[idx] is None]
    retried: set[int] = set(retry_indices)
    if retry_indices:
        retry_prompts = [dataset_module.build_judge_prompt(records[i]) for i in retry_indices]
        retry_outputs = evaluator.predict_batch(retry_prompts, temperature=0.0, max_tokens=16)
        for idx, out in zip(retry_indices, retry_outputs):
            labels[idx] = dataset_module.parse_judge_output(out)

    return labels, retried


# ---------------------------------------------------------------------------
# Record selection
# ---------------------------------------------------------------------------

def _select_records_to_label(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select greedy records for labeling."""
    return [
        r for r in records
        if (r.get("generation_mode") or r.get("metadata", {}).get("generation_mode")) != "sampled"
    ]
