"""Bootstrap confidence intervals for UQ metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.evaluate.metrics import DEFAULT_COVERAGES, evaluate_score


def percentile_ci(values: Sequence[float], ci: float = 0.95) -> tuple[float | None, float | None]:
    """Compute percentile confidence interval."""
    valid = [float(v) for v in values if np.isfinite(float(v))]
    if not valid:
        return None, None
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.percentile(np.asarray(valid), [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(low), float(high)


BOOTSTRAP_METRIC_KEYS = ["auroc", "auprc", "prr", "ece", "brier", "spearman_rho", "kendall_tau"]


def bootstrap_metrics(
    y_error: Sequence[int],
    scores: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> dict[str, Any]:
    """Compute bootstrap CI for all evaluation metrics."""
    labels = np.array(y_error, dtype=np.int64)
    score_arr = np.array(scores, dtype=np.float64)
    n = len(labels)
    rng = np.random.default_rng(seed)

    metric_samples: dict[str, list[float]] = {k: [] for k in BOOTSTRAP_METRIC_KEYS}
    valid_count = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_labels = labels[idx].tolist()
        boot_scores = score_arr[idx].tolist()
        if len(set(boot_labels)) < 2:
            continue
        metrics = evaluate_score(boot_labels, boot_scores, coverages=coverages)
        valid_count += 1
        for key in metric_samples:
            val = metrics.get(key)
            if val is not None and np.isfinite(val):
                metric_samples[key].append(val)

    result: dict[str, Any] = {"n_bootstrap": n_bootstrap, "valid_bootstrap": valid_count}
    for key, samples in metric_samples.items():
        arr = np.array(samples, dtype=np.float64) if samples else np.array([])
        result[f"{key}_bootstrap_mean"] = float(arr.mean()) if arr.size else None
        result[f"{key}_bootstrap_std"] = float(arr.std(ddof=1)) if arr.size > 1 else None
        result[f"{key}_bootstrap_valid"] = int(arr.size)
        lo, hi = percentile_ci(samples)
        result[f"{key}_ci_low"] = lo
        result[f"{key}_ci_high"] = hi
    return result
