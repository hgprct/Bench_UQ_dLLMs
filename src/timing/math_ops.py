"""Standalone per-feature math kernels for latency profiling.

Each timing function returns (elapsed_seconds, value). Re-implementing the
kernels rather than calling compute_sampling_features lets us attribute math
cost to each feature individually under the standalone-cost model.

Numerical results must match src/features/sampling.py and src/features/token.py
exactly; tests assert equivalence.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import numpy as np

from src.features.sampling import _greedy_semantic_clusters, _von_neumann_entropy
from src.features.token import visible_positions, _softmax_entropy, _safe_exp
from src.timing.clock import Timer


# Token-level features (CPU-only, no GPU sync needed)

def time_msp(logprobs: np.ndarray, *, vis: list[int]) -> tuple[float, float | None]:
    with Timer(device=None, sync=False) as t:
        if not vis:
            val = None
        else:
            lp = np.asarray(logprobs, dtype=np.float64)[vis]
            finite = np.isfinite(lp)
            val = _safe_exp(float(lp[finite].sum())) if finite.any() else None
    return float(t.elapsed), val


def time_perplexity(logprobs: np.ndarray, *, vis: list[int]) -> tuple[float, float | None]:
    with Timer(device=None, sync=False) as t:
        if not vis:
            val = None
        else:
            lp = np.asarray(logprobs, dtype=np.float64)[vis]
            finite = np.isfinite(lp)
            val = float(_safe_exp(-lp[finite].mean())) if finite.any() else None
    return float(t.elapsed), val


def time_mte(topk_logits: np.ndarray, *, vis: list[int]) -> tuple[float, float | None]:
    with Timer(device=None, sync=False) as t:
        if not vis or topk_logits.ndim == 1:
            val = None
        else:
            ent = _softmax_entropy(np.asarray(topk_logits, dtype=np.float64)[vis])
            finite = np.isfinite(ent)
            val = float(ent[finite].mean()) if finite.any() else None
    return float(t.elapsed), val


def time_mcnse(
    logprobs_per_item: Sequence[Any],
    token_ids_per_item: Sequence[Any],
    *,
    special_ids: set[int],
    eos_ids: set[int],
) -> tuple[float, float | None]:
    with Timer(device=None, sync=False) as t:
        means = []
        for lp, tids in zip(logprobs_per_item, token_ids_per_item):
            vis = visible_positions(tids, special_ids=special_ids, eos_ids=eos_ids)
            if not vis:
                continue
            arr = np.asarray(lp, dtype=np.float64)[vis]
            finite = np.isfinite(arr)
            if finite.any():
                means.append(float(-arr[finite].mean()))
        val = float(np.mean(means)) if means else None
    return float(t.elapsed), val


# NLI-graph features: standalone attribution.
# Each feature pays for the parts of compute_sampling_features it actually needs.

def _laplacian_eigvals(sym_entail: np.ndarray) -> np.ndarray:
    weights = np.maximum(sym_entail, 0.0).astype(np.float64)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    laplacian = np.diag(degree) - weights
    eig = np.linalg.eigvalsh(0.5 * (laplacian + laplacian.T))
    return np.maximum(eig, 0.0)


def time_se_marginal(sym_entail: np.ndarray) -> tuple[float, float | None]:
    n = sym_entail.shape[0]
    with Timer(device=None, sync=False) as t:
        if n < 2:
            val = None
        else:
            classes = _greedy_semantic_clusters(sym_entail, threshold=0.5)
            num_classes = max(classes) + 1 if classes else 1
            class_counts = Counter(classes)
            probs = np.array(
                [class_counts.get(c, 0) / n for c in range(num_classes)],
                dtype=np.float64,
            )
            probs = probs[probs > 0]
            val = float(-np.sum(probs * np.log(probs))) if probs.size > 1 else 0.0
    return float(t.elapsed), val


def time_se_conditional(sym_entail: np.ndarray) -> tuple[float, float | None]:
    n = sym_entail.shape[0]
    with Timer(device=None, sync=False) as t:
        if n < 2:
            val = None
        else:
            classes = _greedy_semantic_clusters(sym_entail, threshold=0.5)
            class_counts = Counter(classes)
            h_x_given_c = sum(
                (c / n) * math.log(c) for c in class_counts.values() if c > 1
            )
            val = float(h_x_given_c / math.log(n))
    return float(t.elapsed), val


def time_ecc(sym_entail: np.ndarray) -> tuple[float, float | None]:
    n = sym_entail.shape[0]
    with Timer(device=None, sync=False) as t:
        if n < 2:
            val = None
        else:
            eig = _laplacian_eigvals(sym_entail)
            val = float(eig[1]) if eig.size > 1 else None
    return float(t.elapsed), val


def time_eigval(sym_entail: np.ndarray) -> tuple[float, float | None]:
    n = sym_entail.shape[0]
    with Timer(device=None, sync=False) as t:
        if n < 2:
            val = None
        else:
            eig = _laplacian_eigvals(sym_entail)
            val = float((n - np.sum(eig)) / n) if eig.size > 0 else None
    return float(t.elapsed), val


def time_kle_heat(sym_entail: np.ndarray) -> tuple[float, float | None]:
    n = sym_entail.shape[0]
    with Timer(device=None, sync=False) as t:
        if n < 2:
            val = None
        else:
            eig = _laplacian_eigvals(sym_entail)
            factors = np.exp(-0.3 * eig)
            val = float(_von_neumann_entropy(factors))
    return float(t.elapsed), val


def time_kle_matern(sym_entail: np.ndarray) -> tuple[float, float | None]:
    n = sym_entail.shape[0]
    with Timer(device=None, sync=False) as t:
        if n < 2:
            val = None
        else:
            eig = _laplacian_eigvals(sym_entail)
            shift = 2.0
            factors = np.power(np.maximum(shift + eig, 1e-12), -1.0)
            val = float(_von_neumann_entropy(factors))
    return float(t.elapsed), val
