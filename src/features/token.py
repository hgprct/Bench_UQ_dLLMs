"""Token-level UQ features from logprobs and top-k logits.

Features: msp, perplexity, mte (from a single sample's final step),
and mcnse (from multiple items — iid samples or trajectory steps).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def compute_msp(
    logprobs: Sequence[float],
    *,
    visible_positions: list[int],
) -> float | None:
    """Mean Sequence Probability: product of visible-token commit probabilities.

    Stored as confidence (higher = more confident). The evaluator negates it.
    """
    if not visible_positions:
        return None
    lp = np.asarray(logprobs, dtype=np.float64)
    visible_lp = lp[visible_positions]
    finite_mask = np.isfinite(visible_lp)
    if not finite_mask.any():
        return None
    return _safe_exp(float(visible_lp[finite_mask].sum()))


def compute_perplexity(
    logprobs: Sequence[float],
    *,
    visible_positions: list[int],
) -> float | None:
    """Perplexity: exp(mean negative log-prob) over visible tokens."""
    if not visible_positions:
        return None
    lp = np.asarray(logprobs, dtype=np.float64)
    visible_lp = lp[visible_positions]
    finite_mask = np.isfinite(visible_lp)
    if not finite_mask.any():
        return None
    mean_neg_lp = -visible_lp[finite_mask].mean()
    return float(_safe_exp(mean_neg_lp))


def compute_mte(
    topk_logits: np.ndarray,
    *,
    visible_positions: list[int],
) -> float | None:
    """Mean Token Entropy of final-step top-k logits over visible positions."""
    if not visible_positions:
        return None
    logits = np.asarray(topk_logits, dtype=np.float64)
    if logits.ndim == 1:
        return None
    visible_logits = logits[visible_positions]
    entropies = _softmax_entropy(visible_logits)
    finite_mask = np.isfinite(entropies)
    if not finite_mask.any():
        return None
    return float(entropies[finite_mask].mean())


def compute_mcnse(
    logprobs_per_item: Sequence[Any],
    token_ids_per_item: Sequence[Any],
    *,
    special_ids: set[int],
    eos_ids: set[int],
) -> float | None:
    """Monte Carlo Normalized Sequence Entropy: mean normalized neg-log-prob across items."""
    mean_neg_lps: list[float] = []
    for lp, tids in zip(logprobs_per_item, token_ids_per_item):
        vis = visible_positions(tids, special_ids=special_ids, eos_ids=eos_ids)
        if not vis:
            continue
        lp_arr = np.asarray(lp, dtype=np.float64)
        visible_lp = lp_arr[vis]
        finite = np.isfinite(visible_lp)
        if not finite.any():
            continue
        mean_neg_lps.append(float(-visible_lp[finite].mean()))
    if not mean_neg_lps:
        return None
    return float(np.mean(mean_neg_lps))


def visible_positions(
    token_ids: Sequence[Any],
    *,
    special_ids: set[int],
    eos_ids: set[int],
) -> list[int]:
    """Find non-special, non-EOS token positions (stop at first EOS)."""
    positions: list[int] = []
    for idx, tid in enumerate(np.asarray(token_ids).reshape(-1).tolist()):
        try:
            token = int(tid)
        except (TypeError, ValueError):
            continue
        if token in eos_ids:
            break
        if token in special_ids:
            continue
        positions.append(idx)
    return positions


def special_ids_from_config(config: dict[str, Any]) -> tuple[set[int], set[int]]:
    """Extract special and EOS token ID sets from a run config."""
    eos_ids = _as_token_id_set(
        config.get("eos_token_ids"),
        config.get("eot_token_ids"),
        config.get("eos_eot_token_ids"),
    )
    special = set(eos_ids)
    special.update(_as_token_id_set(config.get("mask_id"), config.get("pad_token_id")))
    return special, eos_ids


def _as_token_id_set(*values: Any) -> set[int]:
    out: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            for item in value:
                out.update(_as_token_id_set(item))
            continue
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _safe_exp(value: float) -> float:
    if value < -745.0:
        return 0.0
    if value > 709.0:
        return float("inf")
    return float(math.exp(value))


def _softmax_entropy(logits: np.ndarray) -> np.ndarray:
    """Compute entropy from logits along the last axis."""
    matrix = np.asarray(logits, dtype=np.float64)
    if matrix.size == 0:
        return np.asarray([], dtype=np.float64)
    max_vals = np.nanmax(matrix, axis=-1, keepdims=True)
    shifted = matrix - max_vals
    exp_vals = np.exp(shifted)
    normalizers = exp_vals.sum(axis=-1, keepdims=True)
    probs = np.divide(exp_vals, normalizers, out=np.zeros_like(exp_vals), where=normalizers > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.log(probs)
    entropy = -np.nansum(probs * log_probs, axis=-1)
    return entropy.astype(np.float64)
