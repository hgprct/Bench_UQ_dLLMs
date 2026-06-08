"""Evaluation runner: load features, compute metrics, write outputs."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluate.bootstrap import bootstrap_metrics
from src.evaluate.metrics import (
    error_labels_from_records,
    evaluate_score,
    orient_scores,
)
from src.features.directions import (
    FEATURE_DIRECTIONS,
    all_feature_names,
    build_feature_directions,
)
from src.io.json_utils import read_jsonl, write_json

_INDEXED_RANDOM_RE = re.compile(r"^random-(\d+)-(.+)$")


def _discover_indexed_random_features(records: list[dict]) -> list[str]:
    """Find random-{i}-{base} feature keys present in records."""
    found: set[str] = set()
    for r in records:
        for k in r:
            if _INDEXED_RANDOM_RE.match(k):
                found.add(k)
    return sorted(found)


def _aggregate_indexed_random_metrics(
    features: dict[str, Any],
) -> None:
    """Average metrics across indexed random reps, store as random-{base}.

    Modifies *features* in place: removes random-{i}-{base} entries,
    adds random-{base} with averaged metrics.
    """
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for fname, metrics in list(features.items()):
        m = _INDEXED_RANDOM_RE.match(fname)
        if m:
            rep_idx = int(m.group(1))
            base = m.group(2)
            groups[base].append((rep_idx, metrics))

    if not groups:
        return

    for base, reps in groups.items():
        for rep_idx, _ in reps:
            del features[f"random-{rep_idx}-{base}"]

        all_keys: set[str] = set()
        for _, metrics in reps:
            all_keys.update(metrics.keys())

        metric_values: dict[str, list[float]] = defaultdict(list)
        for _, metrics in sorted(reps):
            for key in all_keys:
                val = metrics.get(key)
                if val is None:
                    continue
                try:
                    metric_values[key].append(float(val))
                except (TypeError, ValueError):
                    pass

        averaged: dict[str, Any] = {}
        for key in all_keys:
            vals = metric_values.get(key)
            if vals:
                averaged[key] = sum(vals) / len(vals)
            else:
                averaged[key] = None
        averaged["n_random_selections"] = len(reps)
        features[f"random-{base}"] = averaged


def evaluate_run(
    features_path: str | Path,
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    bootstrap: bool = True,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    report_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate all UQ features and write results."""
    records = read_jsonl(features_path)
    if not records:
        raise ValueError(f"No feature records found in {features_path}")

    n_ambiguous = sum(1 for r in records if r.get("is_correct") is None)
    if n_ambiguous:
        print(
            f"[WARNING] {n_ambiguous}/{len(records)} records have is_correct=None "
            f"(e.g. ambiguous LLM judge); excluding from metric computation."
        )
    records = [r for r in records if r.get("is_correct") is not None]
    if not records:
        raise ValueError(f"All feature records in {features_path} have is_correct=None")

    directions = build_feature_directions()
    feature_names = [k for k in all_feature_names() if any(k in r for r in records)]

    indexed_random = _discover_indexed_random_features(records)
    if indexed_random:
        for fname in indexed_random:
            m = _INDEXED_RANDOM_RE.match(fname)
            if m:
                base = m.group(2)
                if base in FEATURE_DIRECTIONS:
                    directions[fname] = FEATURE_DIRECTIONS[base]
        feature_names.extend(indexed_random)

    y_error = error_labels_from_records(records)
    n_correct = sum(1 for e in y_error if e == 0)
    n_examples = len(records)

    token_counts = [r["n_visible_tokens"] for r in records if r.get("n_visible_tokens") is not None]
    mean_visible_tokens = sum(token_counts) / len(token_counts) if token_counts else None

    results: dict[str, Any] = {
        "n_examples": n_examples,
        "n_correct": n_correct,
        "n_wrong": n_examples - n_correct,
        "accuracy": n_correct / n_examples if n_examples > 0 else None,
        "mean_visible_tokens": mean_visible_tokens,
        "features": {},
    }

    for feature_name in feature_names:
        raw_scores = [float(r.get(feature_name) if r.get(feature_name) is not None else 0.0) for r in records]
        scores = orient_scores(raw_scores, feature_name, directions)

        valid_mask = [r.get(feature_name) is not None for r in records]
        valid_labels = [y_error[i] for i, v in enumerate(valid_mask) if v]
        valid_scores = [scores[i] for i, v in enumerate(valid_mask) if v]

        if len(set(valid_labels)) < 2 or len(valid_labels) < 2:
            results["features"][feature_name] = {
                "auroc": None, "auprc": None, "prr": None,
                "ece": None, "brier": None, "spearman_rho": None, "kendall_tau": None,
            }
            continue

        metrics = evaluate_score(valid_labels, valid_scores)
        if bootstrap:
            boot = bootstrap_metrics(valid_labels, valid_scores,
                                     n_bootstrap=bootstrap_samples, seed=bootstrap_seed)
            metrics.update(boot)
        results["features"][feature_name] = metrics

    _aggregate_indexed_random_metrics(results["features"])

    if output_json:
        write_json(output_json, results)
    if output_csv:
        _write_csv(output_csv, results)
    return results


def _write_csv(path: str | Path, results: dict) -> None:
    """Write a simple CSV summary."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["feature", "auroc", "auprc", "prr", "ece", "brier", "spearman_rho", "kendall_tau"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for name, metrics in results.get("features", {}).items():
            writer.writerow([name] + [metrics.get(c, "") for c in columns[1:]])
