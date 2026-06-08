"""Single-trajectory UQ features: averaged dissimilarity (AD).

AD measures how much intermediate step x0 predictions deviate from the
final answer of the same trajectory, using NLI entailment probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mmd.load import is_valid_text


def compute_step_dissimilarities(
    step_answers: tuple[str, ...],
    final_answer: str,
    entailment_model: Any,
    *,
    nli_batch_size: int = 8,
) -> np.ndarray:
    """NLI dissimilarity between each step answer and the final answer.

    Returns shape (T,) with dissim[t] = 1 - sym_entail(step_t, final).
    Invalid or identical-to-final steps get 0.
    """
    T = len(step_answers)
    dissim = np.zeros(T, dtype=np.float64)

    if not is_valid_text(final_answer):
        return dissim

    final_text = final_answer.strip()
    entail_id = int(getattr(entailment_model, "entailment_id", 2))

    valid_indices: list[int] = []
    valid_texts: list[str] = []
    for t in range(T):
        text = step_answers[t]
        if is_valid_text(text):
            text = text.strip()
            if text != final_text:
                valid_indices.append(t)
                valid_texts.append(text)

    if not valid_texts:
        return dissim

    premises: list[str] = []
    hypotheses: list[str] = []
    for text in valid_texts:
        premises.append(text)
        hypotheses.append(final_text)
        premises.append(final_text)
        hypotheses.append(text)

    probs, _ = entailment_model.batch_probabilities(
        premises, hypotheses, batch_size=nli_batch_size,
    )
    probs = np.asarray(probs, dtype=np.float64)

    for i, t in enumerate(valid_indices):
        p_forward = probs[2 * i, entail_id]
        p_backward = probs[2 * i + 1, entail_id]
        dissim[t] = 1.0 - (p_forward + p_backward) / 2.0

    return dissim


def compute_averaged_dissimilarity(
    step_answers: tuple[str, ...],
    final_answer: str,
    entailment_model: Any,
    *,
    weights: np.ndarray | None = None,
    nli_batch_size: int = 8,
) -> float | None:
    """Averaged dissimilarity between step predictions and final answer.

    weights=None  → full-trajectory: (1/T) * sum(dissim).
    weights given → selection:       dot(weights, dissim).
    """
    if not step_answers or not is_valid_text(final_answer):
        return None

    dissim = compute_step_dissimilarities(
        step_answers, final_answer, entailment_model,
        nli_batch_size=nli_batch_size,
    )

    T = len(dissim)
    if weights is not None:
        if len(weights) != T:
            return None
        return float(np.dot(weights, dissim))
    return float(np.sum(dissim) / T)
