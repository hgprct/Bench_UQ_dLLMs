"""Feature builder orchestrator: loads run data and computes all UQ features.

Produces uq_features.jsonl with one row per original prompt containing
base features (msp, perplexity, mte) and sampling features (mcnse, se-marginal,
se-conditional, ecc, eigval, kle-*).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.features.directions import (
    BASE_FEATURES,
    SAMPLING_FEATURES,
)
from src.features.sampling import CachedEntailmentModel, compute_sampling_features
from src.features.token import (
    compute_mcnse,
    compute_msp,
    compute_mte,
    compute_perplexity,
    special_ids_from_config,
    visible_positions,
)
from src.utils.io import read_json, read_jsonl


def build_feature_rows(
    run_dir: str | Path,
    *,
    nli_model: Any,
    nli_batch_size: int = 64,
    prompt_batch_size: int = 0,
    sample_nli_max_items: int | None = None,
    progress_every: int = 10,
) -> list[dict[str, Any]]:
    """Build one feature row per original prompt from a generation run."""
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")

    records = read_jsonl(run_dir / "answers.jsonl")

    topk_path = run_dir / "topk_logprobs.npz"
    topk = dict(np.load(topk_path, allow_pickle=False)) if topk_path.exists() else {}

    special, eos = special_ids_from_config(config)
    cached_nli = CachedEntailmentModel(nli_model) if nli_model is not None else None

    grouped = _group_by_prompt(records, config)
    prompt_items = list(grouped.items())
    effective_batch = len(prompt_items) if prompt_batch_size <= 0 else prompt_batch_size

    rows = []
    for batch_start in range(0, len(prompt_items), effective_batch):
        batch = prompt_items[batch_start:batch_start + effective_batch]

        if cached_nli:
            _prefetch_nli_batch(batch, cached_nli, nli_batch_size, sample_nli_max_items)

        for offset, (prompt_id, group) in enumerate(batch):
            prompt_idx = batch_start + offset
            greedy, sampled = _partition_greedy_sampled(group)
            row: dict[str, Any] = {
                "prompt_id": prompt_id,
                "is_correct": greedy.get("label") if greedy else None,
            }

            if greedy and topk:
                sample_idx = greedy.get("sample_id", 0)
                final_logprobs = _extract_logprobs(topk, sample_idx)
                final_topk = _extract_topk_logprobs(topk, sample_idx)
                vis_pos = list(range(final_logprobs.shape[0])) if final_logprobs is not None else []
                row["n_visible_tokens"] = len(vis_pos)
                row["msp"] = compute_msp(final_logprobs, visible_positions=vis_pos) if final_logprobs is not None else None
                row["perplexity"] = compute_perplexity(final_logprobs, visible_positions=vis_pos) if final_logprobs is not None else None
                row["mte"] = compute_mte(final_topk, visible_positions=vis_pos) if final_topk is not None else None
            else:
                row["n_visible_tokens"] = None
                for f in BASE_FEATURES:
                    row[f] = None

            for f in SAMPLING_FEATURES:
                row[f] = None

            if cached_nli and sampled:
                texts = [_final_answer(r) for r in sampled if _final_answer(r)]
                nli_feats = compute_sampling_features(texts, cached_nli, nli_batch_size=nli_batch_size, max_items=sample_nli_max_items)
                row.update(nli_feats)

            if sampled and topk:
                sample_lps = _extract_sampled_logprobs(topk, sampled)
                if sample_lps:
                    row["mcnse"] = compute_mcnse(sample_lps, [None] * len(sample_lps), special_ids=special, eos_ids=eos)

            rows.append(row)
            if progress_every and (prompt_idx + 1) % progress_every == 0:
                print(f"[features] {prompt_idx + 1}/{len(prompt_items)} prompts processed")

    return rows


def _prefetch_nli_batch(
    batch: list[tuple[str, list[dict]]],
    cached_nli: CachedEntailmentModel,
    nli_batch_size: int,
    sample_nli_max_items: int | None,
) -> None:
    from src.features.sampling import _subsample_indices

    all_premises: list[str] = []
    all_hypotheses: list[str] = []
    for _, group in batch:
        _, sampled = _partition_greedy_sampled(group)
        if not sampled:
            continue
        texts = [_final_answer(r) for r in sampled if _final_answer(r)]
        if len(texts) < 2:
            continue
        if sample_nli_max_items and len(texts) > sample_nli_max_items:
            indices = _subsample_indices(len(texts), sample_nli_max_items)
            texts = [texts[i] for i in indices]
        for left in texts:
            for right in texts:
                all_premises.append(left)
                all_hypotheses.append(right)

    if all_premises:
        cached_nli.batch_probabilities(all_premises, all_hypotheses, batch_size=nli_batch_size)


def _group_by_prompt(records: list[dict], config: dict) -> dict[str, list[dict]]:
    from collections import OrderedDict
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    num_samples = int(config.get("num_response_samples", 1) or 1)
    for idx, record in enumerate(records):
        pid = record.get("qa_example_id")
        if pid is None:
            records_per_prompt = 1 + num_samples if num_samples > 1 else 1
            pid = str(idx // records_per_prompt)
        grouped.setdefault(str(pid), []).append(record)
    return dict(grouped)


def _partition_greedy_sampled(records: list[dict]) -> tuple[dict | None, list[dict]]:
    greedy = None
    sampled = []
    for record in records:
        mode = record.get("generation_mode")
        if mode == "greedy":
            greedy = record
        else:
            sampled.append(record)
    if greedy is None and records:
        greedy = records[0]
        sampled = records[1:]
    return greedy, sampled


def _final_answer(record: dict) -> str:
    raw = record.get("raw_answer")
    if raw is not None:
        return str(raw).strip()
    final = record.get("final", {})
    return str(final.get("answer", final.get("response", ""))).strip()


def _extract_logprobs(topk: dict, sample_idx: int) -> np.ndarray | None:
    """Extract per-token logprobs (top-1) for a sample."""
    lp = topk.get("topk_logprobs")
    if lp is not None and lp.ndim == 3 and sample_idx < lp.shape[0]:
        return lp[sample_idx, :, 0]
    return None


def _extract_topk_logprobs(topk: dict, sample_idx: int) -> np.ndarray | None:
    """Extract top-k logprobs matrix for a sample."""
    tk = topk.get("topk_logprobs")
    if tk is not None and tk.ndim == 3 and sample_idx < tk.shape[0]:
        return tk[sample_idx]
    return None


def _extract_sampled_logprobs(
    topk: dict,
    sampled_records: list[dict],
) -> list[np.ndarray]:
    logprobs_list: list[np.ndarray] = []
    for r in sampled_records:
        sid = r.get("sample_id")
        if sid is None:
            continue
        lp = _extract_logprobs(topk, int(sid))
        if lp is not None:
            logprobs_list.append(lp)
    return logprobs_list
