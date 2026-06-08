"""NLI-based UQ features from multiple text samples.

Features: se-marginal, se-conditional, ecc, eigval, kle-heat, kle-matern.
These use NLI-derived semantic equivalence graphs over sampled responses.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
import scipy.linalg

KLE_KERNELS = ("heat", "matern")


class CachedEntailmentModel:
    """Memoize exact NLI pairs within one feature-build process."""

    def __init__(self, base_model: Any) -> None:
        self.base_model = base_model
        self._cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def batch_probabilities(
        self,
        text_list1: Sequence[str],
        text_list2: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        keys = [(str(l), str(r)) for l, r in zip(text_list1, text_list2)]
        if not keys:
            return self.base_model.batch_probabilities([], [], batch_size=batch_size)

        unique_missing_keys = set()
        missing_keys, missing_left, missing_right = [], [], []
        for key in keys:
            if key not in self._cache and key not in unique_missing_keys:
                unique_missing_keys.add(key)
                missing_keys.append(key)
                missing_left.append(key[0])
                missing_right.append(key[1])

        if missing_keys:
            probs, classes = self.base_model.batch_probabilities(missing_left, missing_right, batch_size=batch_size)
            for key, prob, cls in zip(missing_keys, np.asarray(probs), np.asarray(classes)):
                self._cache[key] = (np.asarray(prob, dtype=np.float32), int(cls))

        cached_probs = [self._cache[key][0] for key in keys]
        cached_classes = [self._cache[key][1] for key in keys]
        return np.stack(cached_probs, axis=0), np.asarray(cached_classes, dtype=np.int64)


def compute_sampling_features(
    texts: list[str],
    nli_model: Any,
    *,
    nli_batch_size: int = 8,
    max_items: int | None = None,
) -> dict[str, float | None]:
    """Compute NLI-based features from a set of response texts.

    Returns dict with keys: se-marginal, se-conditional, ecc, eigval, kle-heat, kle-matern.
    """
    if len(texts) < 2:
        return {k: None for k in ("se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern")}

    if max_items and len(texts) > max_items:
        indices = _subsample_indices(len(texts), max_items)
        texts = [texts[i] for i in indices]

    entail_id = int(getattr(nli_model, "entailment_id", 2))
    n = len(texts)

    # Build NLI probability matrix
    premises, hypotheses = [], []
    for left in texts:
        for right in texts:
            premises.append(left)
            hypotheses.append(right)
    probs, _ = nli_model.batch_probabilities(premises, hypotheses, batch_size=nli_batch_size)
    probs = np.asarray(probs, dtype=np.float64).reshape(n, n, -1)

    # Entailment matrix
    entail = probs[:, :, entail_id]
    np.fill_diagonal(entail, 1.0)
    sym_entail = (entail + entail.T) / 2.0

    # Semantic equivalence classes via greedy clustering
    classes = _greedy_semantic_clusters(sym_entail, threshold=0.5)
    num_classes = max(classes) + 1 if classes else 1

    # SE-marginal: H(C) — entropy of semantic class distribution
    class_counts = Counter(classes)
    class_probs = np.array([class_counts.get(c, 0) / n for c in range(num_classes)], dtype=np.float64)
    class_probs = class_probs[class_probs > 0]
    se_marginal = float(-np.sum(class_probs * np.log(class_probs))) if class_probs.size > 1 else 0.0

    # SE-conditional: H(X|C)/log(n) — within-cluster agreement (1=all identical, 0=all singletons)
    if n > 1:
        h_x_given_c = sum(
            (count / n) * np.log(count)
            for count in class_counts.values() if count > 1
        )
        se_conditional = float(h_x_given_c / np.log(n))
    else:
        se_conditional = 0.0

    # Graph Laplacian features
    weights = np.maximum(sym_entail, 0.0).astype(np.float64)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=1)
    laplacian = np.diag(degree) - weights

    eigenvalues = np.linalg.eigvalsh(0.5 * (laplacian + laplacian.T))
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # ECC: algebraic connectivity (Fiedler value = second smallest Laplacian eigenvalue)
    ecc = float(eigenvalues[1]) if eigenvalues.size > 1 else None

    # Eigval sum
    eigval = float((n - np.sum(eigenvalues)) / n) if eigenvalues.size > 0 else None

    # KLE kernels
    kle_features = {}
    for kernel_name in KLE_KERNELS:
        if kernel_name == "heat":
            factors = np.exp(-0.3 * eigenvalues)
        else:
            shift = 2.0
            factors = np.power(np.maximum(shift + eigenvalues, 1e-12), -1.0)
        kle_features[f"kle-{kernel_name}"] = float(_von_neumann_entropy(factors))

    return {
        "se-marginal": se_marginal,
        "se-conditional": se_conditional,
        "ecc": ecc,
        "eigval": eigval,
        **kle_features,
    }


def _greedy_semantic_clusters(entailment_matrix: np.ndarray, threshold: float = 0.5) -> list[int]:
    """Assign texts to semantic equivalence classes via greedy clustering."""
    n = entailment_matrix.shape[0]
    classes = [-1] * n
    next_class = 0
    for i in range(n):
        if classes[i] >= 0:
            continue
        classes[i] = next_class
        for j in range(i + 1, n):
            if classes[j] >= 0:
                continue
            if entailment_matrix[i, j] >= threshold:
                classes[j] = next_class
        next_class += 1
    return classes


def _von_neumann_entropy(factors: np.ndarray) -> float:
    """Von Neumann entropy of a kernel defined by spectral factors."""
    factors = np.asarray(factors, dtype=np.float64)
    factors = factors[factors > 1e-12]
    if factors.size == 0:
        return 0.0
    factors = factors / factors.sum()
    return float(-np.sum(factors * np.log(factors)))


def _subsample_indices(num_items: int, max_items: int) -> list[int]:
    """Evenly-spaced subsample indices."""
    if num_items <= max_items:
        return list(range(num_items))
    raw = np.linspace(0, num_items - 1, num=max_items)
    return sorted({int(round(v)) for v in raw})
