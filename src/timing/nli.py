"""Timed NLI calls that bypass the per-prompt cache used in the main pipeline.

For honest latency we want each prompt to pay the full N×N inference cost,
not benefit from any prior prompt's cached pairs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.timing.clock import Timer


def all_pairs_entail_matrix(
    nli_model: Any,
    texts: list[str],
    *,
    nli_batch_size: int,
    device: Any,
    entail_id: int = 2,
) -> tuple[float, np.ndarray, int]:
    """Run NLI on all N×N (premise, hypothesis) pairs.

    Returns (elapsed_seconds, symmetrized_entailment_matrix [N,N], n_pairs).
    The matrix mirrors compute_sampling_features: diagonal forced to 1, then
    averaged with its transpose.
    """
    n = len(texts)
    if n < 2:
        return 0.0, np.zeros((n, n), dtype=np.float64), 0

    premises, hypotheses = [], []
    for left in texts:
        for right in texts:
            premises.append(left)
            hypotheses.append(right)

    with Timer(device) as t:
        probs, _ = nli_model.batch_probabilities(premises, hypotheses, batch_size=nli_batch_size)
    probs = np.asarray(probs, dtype=np.float64).reshape(n, n, -1)
    entail = probs[:, :, entail_id]
    np.fill_diagonal(entail, 1.0)
    sym_entail = (entail + entail.T) / 2.0

    return float(t.elapsed), sym_entail, n * n
