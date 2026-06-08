"""K-fold cross-validation aggregation: mean/std across folds."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from src.io.json_utils import to_jsonable, write_json


def aggregate_kfold_results(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute mean and std of metrics across folds.

    Each entry in *fold_results* has keys 'baseline' and 'selection',
    each mapping to an evaluate_run-style dict with a 'features' sub-dict.
    """
    n_folds = len(fold_results)
    summary: dict[str, Any] = {"n_folds": n_folds, "baseline": {}, "selection": {}}

    for section in ("baseline", "selection"):
        all_features: dict[str, dict[str, list[float]]] = {}
        for fold in fold_results:
            features = fold.get(section, {}).get("features", {})
            for feat_name, metrics in features.items():
                if feat_name not in all_features:
                    all_features[feat_name] = {}
                for metric_name, value in metrics.items():
                    if not _is_numeric(value):
                        continue
                    all_features[feat_name].setdefault(metric_name, []).append(float(value))

        for feat_name in sorted(all_features):
            summary[section][feat_name] = {}
            for metric_name, values in sorted(all_features[feat_name].items()):
                clean = [v for v in values if math.isfinite(v)]
                if not clean:
                    continue
                mean = sum(clean) / len(clean)
                std = (sum((v - mean) ** 2 for v in clean) / len(clean)) ** 0.5
                summary[section][feat_name][metric_name] = {
                    "mean": mean,
                    "std": std,
                    "per_fold": values,
                }

    return summary


def write_kfold_summary(
    summary: dict[str, Any],
    fold_results: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full = {}
    if meta:
        full.update(meta)
    full.update(summary)
    full["per_fold"] = [
        {
            "fold": i,
            "test_size": fr.get("test_size"),
            "best_lambda": fr.get("best_lambda"),
            "selected_steps": fr.get("selected_steps"),
        }
        for i, fr in enumerate(fold_results)
    ]

    write_json(output_dir / "kfold_summary.json", to_jsonable(full))
    _write_summary_csv(output_dir / "kfold_summary.csv", summary)


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    n_folds = summary.get("n_folds", 0)
    fold_headers = [f"fold_{i}" for i in range(n_folds)]
    columns = ["type", "feature", "metric", "mean", "std"] + fold_headers

    rows: list[list[Any]] = []
    for section in ("baseline", "selection"):
        for feat_name, metrics in sorted(summary.get(section, {}).items()):
            for metric_name, vals in sorted(metrics.items()):
                row = [
                    section, feat_name, metric_name,
                    vals.get("mean"), vals.get("std"),
                ]
                row.extend(vals.get("per_fold", []))
                rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(v) for v in row])


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _is_numeric(value: Any) -> bool:
    if value is None:
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
