"""Core evaluation metrics: AUROC, AUPRC, PRR, ECE, Brier, Spearman, Kendall."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.directions import FEATURE_DIRECTIONS

DEFAULT_COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
DEFAULT_ECE_BINS = 10


def selective_risk_curve(
    y_error: Sequence[int],
    scores: Sequence[float],
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> list[dict[str, Any]]:
    """Compute selective risk at each coverage level."""
    n = len(y_error)
    if n == 0:
        return []
    sorted_idx = np.argsort(scores)
    sorted_errors = np.array(y_error)[sorted_idx]
    results = []
    for coverage in coverages:
        k = max(1, int(round(coverage * n)))
        selected = sorted_errors[:k]
        risk = float(selected.mean())
        results.append({
            "coverage": float(coverage),
            "k": k,
            "selective_risk": risk,
            "selective_accuracy": 1.0 - risk,
        })
    return results


def prr(
    y_error: Sequence[int],
    scores: Sequence[float],
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> float | None:
    """Prediction Rejection Ratio."""
    n = len(y_error)
    if n < 2:
        return None
    errors = np.array(y_error, dtype=np.float64)
    base_error = errors.mean()
    if base_error <= 0 or base_error >= 1:
        return None

    model_curve = selective_risk_curve(y_error, scores, coverages=coverages)
    oracle_scores = list(errors)
    oracle_curve = selective_risk_curve(y_error, oracle_scores, coverages=coverages)
    cov = [r["coverage"] for r in model_curve]
    random_auc = np.trapz([base_error] * len(cov), cov)
    model_auc = np.trapz([r["selective_risk"] for r in model_curve], cov)
    oracle_auc = np.trapz([r["selective_risk"] for r in oracle_curve], [r["coverage"] for r in oracle_curve])

    denom = random_auc - oracle_auc
    if abs(denom) < 1e-12:
        return None
    return float((random_auc - model_auc) / denom)


def minmax_normalize(scores: Sequence[float]) -> np.ndarray:
    """Normalize scores to [0, 1] via min-max scaling."""
    arr = np.asarray(scores, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def expected_calibration_error(
    y_error: Sequence[int],
    scores_01: np.ndarray,
    *,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float | None:
    """ECE with equal-width bins on min-max normalized uncertainty scores."""
    labels = np.asarray(y_error, dtype=np.float64)
    n = len(labels)
    if n < 2:
        return None
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores_01 >= bin_edges[i]) & (scores_01 < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= (scores_01 == bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = float(scores_01[mask].mean())
        avg_error = float(labels[mask].mean())
        ece += (count / n) * abs(avg_error - avg_confidence)
    return float(ece)


def brier_score(
    y_error: Sequence[int],
    scores_01: np.ndarray,
) -> float | None:
    """Brier score: mean squared difference between normalized score and label."""
    labels = np.asarray(y_error, dtype=np.float64)
    if len(labels) < 2:
        return None
    return float(np.mean((scores_01 - labels) ** 2))


def spearman_rho(
    y_error: Sequence[int],
    scores: Sequence[float],
) -> float | None:
    """Spearman's rank correlation between scores and error labels."""
    if len(y_error) < 3 or len(set(y_error)) < 2:
        return None
    rho, _ = spearmanr(scores, y_error)
    if np.isfinite(rho):
        return float(rho)
    return None


def kendall_tau(
    y_error: Sequence[int],
    scores: Sequence[float],
) -> float | None:
    """Kendall's tau rank correlation between scores and error labels."""
    if len(y_error) < 3 or len(set(y_error)) < 2:
        return None
    tau, _ = kendalltau(scores, y_error)
    if np.isfinite(tau):
        return float(tau)
    return None


def evaluate_score(
    y_error: Sequence[int],
    scores: Sequence[float],
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    n_ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, float | None]:
    """Compute AUROC, AUPRC, PRR, ECE, Brier, Spearman, Kendall for one score vector."""
    labels = [int(l) for l in y_error]
    numeric = [float(s) for s in scores]
    has_two = len(set(labels)) == 2 and len(labels) >= 2
    scores_01 = minmax_normalize(numeric)
    return {
        "auroc": float(roc_auc_score(labels, numeric)) if has_two else None,
        "auprc": float(average_precision_score(labels, numeric)) if has_two else None,
        "prr": prr(labels, numeric, coverages=coverages),
        "ece": expected_calibration_error(labels, scores_01, n_bins=n_ece_bins) if has_two else None,
        "brier": brier_score(labels, scores_01) if has_two else None,
        "spearman_rho": spearman_rho(labels, numeric) if has_two else None,
        "kendall_tau": kendall_tau(labels, numeric) if has_two else None,
    }


def orient_scores(
    scores: Sequence[float],
    feature_name: str,
    directions: dict[str, str] | None = None,
) -> list[float]:
    """Negate confidence features so all scores are uncertainty-oriented."""
    dirs = directions or FEATURE_DIRECTIONS
    direction = dirs.get(feature_name, "uncertainty")
    if direction == "confidence":
        return [-float(s) for s in scores]
    return [float(s) for s in scores]


def error_labels_from_records(records: Sequence[dict]) -> list[int]:
    """Extract binary error labels (1=wrong, 0=correct) from feature rows.

    Callers must filter out records with is_correct=None first.
    """
    labels = []
    for record in records:
        is_correct = record.get("is_correct")
        if is_correct is None:
            raise ValueError("Record with is_correct=None; filter ambiguous records before calling")
        labels.append(0 if bool(is_correct) else 1)
    return labels
