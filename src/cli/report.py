"""CLI entry point: unified results report.

Produces:
  - XLSX workbook with one tab per model (performance metrics only)
  - Pareto frontier plots per (model, dataset, remasking) config
    merging performance from eval metrics with timing from latency profiles.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export UQ results: XLSX workbook + Pareto frontier plots.",
    )
    parser.add_argument("--outputs_root", action="append", default=None,
                        help="Root directory to scan for uq_eval_metrics.json (repeat for multiple). Default: outputs")
    parser.add_argument("--latency_dir", default=None,
                        help="Directory containing per-config latency profile subdirs (e.g. outputs/latency)")
    parser.add_argument("--output_dir", default="outputs/report",
                        help="Output directory for XLSX and plots")
    parser.add_argument("--metric", default="prr",
                        help="Performance metric to report (default: prr)")
    parser.add_argument("--metrics_name", default="uq_eval_metrics.json",
                        help="Metrics JSON filename to scan for")
    parser.add_argument("--format", dest="plot_format", default="png",
                        choices=["pdf", "png", "svg"],
                        help="Plot output format")
    parser.add_argument("--no_xlsx", action="store_true",
                        help="Skip XLSX generation")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip Pareto plot generation")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Latency loading
# ---------------------------------------------------------------------------

def _load_latency_csv(path: Path) -> dict[str, dict[str, float]]:
    latency: dict[str, dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = row["feature"]
            latency[feat] = {
                "overhead_mean": float(row.get("overhead_mean", 0.0)),
                "overhead_std": float(row.get("overhead_std", 0.0)),
                "time_total_mean": float(row.get("time_total_mean", 0.0)),
                "time_total_std": float(row.get("time_total_std", 0.0)),
                "time_gen_mean": float(row.get("time_gen_mean", 0.0)),
                "time_nli_mean": float(row.get("time_nli_mean", 0.0)),
                "time_math_mean": float(row.get("time_math_mean", 0.0)),
                "family": row.get("family", ""),
            }
    return latency


def _load_latency_metadata(meta_path: Path) -> dict[str, Any]:
    from src.io.json_utils import read_json
    return read_json(meta_path)


def _discover_latency(latency_dir: Path) -> dict[tuple[str, str, str], tuple[dict, Path]]:
    """Discover latency profiles, keyed by (model_id, dataset, remasking).

    Searches recursively so latency data can live in a flat directory
    (outputs/latency/<config>/) or inside run dirs (<run>/latency/).
    """
    from src.io.json_utils import read_json
    from src.io.results_xlsx import _resolve_model_name

    found: dict[tuple[str, str, str], tuple[dict, Path]] = {}
    if not latency_dir.is_dir():
        return found

    for csv_path in sorted(latency_dir.rglob("latency_summary.csv")):
        meta_path = csv_path.parent / "latency_metadata.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        model_name = _resolve_model_name(meta.get("model_id", ""))
        dataset = meta.get("dataset", "")
        remasking = meta.get("remasking", "")
        if model_name and dataset and remasking:
            latency_data = _load_latency_csv(csv_path)
            found[(model_name, dataset, remasking)] = (latency_data, csv_path)

    return found


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def _compute_pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Return indices on the Pareto frontier (minimize x, maximize y)."""
    if not points:
        return []
    indexed = sorted(enumerate(points), key=lambda t: (t[1][0], -t[1][1]))
    front = []
    best_y = float("-inf")
    for idx, (x, y) in indexed:
        if y > best_y:
            front.append(idx)
            best_y = y
    return front


FAMILY_STYLES = {
    "token": {"color": "#2ECC71", "marker": "o", "label": "Token"},
    "iid": {"color": "#3498DB", "marker": "s", "label": "IID sample"},
    "full": {"color": "#9B59B6", "marker": "D", "label": "Full trajectory"},
    "selected": {"color": "#E74C3C", "marker": "^", "label": "Selected step"},
}


def _feature_family(feat_name: str) -> str:
    if feat_name.startswith("selected-") or feat_name == "weighted-ad":
        return "selected"
    if feat_name.startswith("full-"):
        return "full"
    if feat_name.startswith("random-"):
        return "random"
    if feat_name in ("msp", "perplexity", "mte"):
        return "token"
    return "iid"


def _write_pareto(
    perf_data: dict[str, float],
    latency_data: dict[str, dict[str, float]],
    output_path: Path,
    *,
    metric: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    points: list[tuple[float, float, float, float, str, str]] = []
    for feat, lat in latency_data.items():
        if feat not in perf_data:
            continue
        perf_val = perf_data[feat]
        if perf_val is None:
            continue
        overhead = lat["overhead_mean"]
        overhead_std = lat.get("overhead_std", 0.0)
        family = _feature_family(feat)
        points.append((overhead, perf_val, overhead_std, 0.0, feat, family))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    families_plotted: set[str] = set()
    for overhead, perf_val, oh_std, _, feat, family in points:
        style = FAMILY_STYLES.get(family, {"color": "#95A5A6", "marker": "x", "label": family})
        label = style["label"] if family not in families_plotted else None
        families_plotted.add(family)
        ax.errorbar(
            overhead, perf_val, xerr=oh_std,
            fmt=style["marker"], c=style["color"], ms=8, capsize=3,
            zorder=3, label=label, markeredgecolor="white", markeredgewidth=0.5,
        )
        ax.annotate(feat, (overhead, perf_val), fontsize=6.5,
                    xytext=(5, 5), textcoords="offset points", alpha=0.75)

    xy = [(p[0], p[1]) for p in points]
    front = _compute_pareto_front(xy)
    if len(front) >= 2:
        front_sorted = sorted(front, key=lambda i: xy[i][0])
        ax.plot([xy[i][0] for i in front_sorted],
                [xy[i][1] for i in front_sorted],
                "k--", alpha=0.4, linewidth=1.5, label="Pareto front")

    ax.set_xlabel("Overhead above baseline (seconds)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[report] Wrote Pareto plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from src.io.results_xlsx import (
        _resolve_model_name,
        _selection_feature_name,
        find_kfold_summaries,
        find_metrics_files,
        load_kfold_results,
        load_run_result,
        write_model_workbook,
    )

    args = parse_args(argv)
    roots = args.outputs_root or ["outputs"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = args.metric.lower()

    # -- Load performance results (prefer kfold summaries over quickeval) --
    kfold_files = find_kfold_summaries(roots)
    kfold_run_dirs: set[Path] = set()
    results: list = []
    for kf in kfold_files:
        kfold_run_dir = kf.parent.parent.resolve()
        kfold_run_dirs.add(kfold_run_dir)
        results.extend(load_kfold_results(kf))
    if kfold_files:
        print(f"[report] Loaded {len(kfold_files)} kfold summary(ies) -> {len(results)} result(s)")

    metric_files = find_metrics_files(roots, metrics_name=args.metrics_name)
    metric_files = [
        f for f in metric_files
        if not any(f.resolve().is_relative_to(d) for d in kfold_run_dirs)
    ]
    if metric_files:
        fallback = [load_run_result(path) for path in metric_files]
        results.extend(fallback)
        print(f"[report] Loaded {len(fallback)} additional run(s) from {args.metrics_name}")

    if not results:
        print(f"[report] No kfold summaries or {args.metrics_name} files found under: {', '.join(roots)}")
        return
    print(f"[report] Total: {len(results)} result(s)")

    # -- XLSX --
    if not args.no_xlsx:
        xlsx_path = output_dir / "uq_results.xlsx"
        write_model_workbook(results, xlsx_path, metric=metric)

    # -- Pareto plots --
    if not args.no_plots and args.latency_dir:
        latency_dir = Path(args.latency_dir)
        latency_map = _discover_latency(latency_dir)
        if not latency_map:
            print(f"[report] No latency profiles found in {latency_dir}")
        else:
            print(f"[report] Found {len(latency_map)} latency profile(s)")

            perf_by_config: dict[tuple[str, str, str], dict[str, float]] = {}
            for r in results:
                model_name = _resolve_model_name(r.competitor.model_id)
                key = (model_name, r.dataset, r.competitor.remasking)
                feat_metrics = perf_by_config.setdefault(key, {})
                is_selection = r.run_dir.name == "selection"
                for feat, mvals in r.feature_metrics.items():
                    v = mvals.get(metric)
                    if v is None:
                        continue
                    if is_selection:
                        feat_metrics[_selection_feature_name(feat)] = float(v)
                    else:
                        feat_metrics.setdefault(feat, float(v))

            n_plots = 0
            for config_key, (lat_data, lat_path) in sorted(latency_map.items()):
                model_name, dataset, remasking = config_key
                if config_key not in perf_by_config:
                    continue
                perf = perf_by_config[config_key]
                if not perf:
                    continue

                plot_name = f"pareto_{model_name}_{dataset}_{remasking}_{metric}.{args.plot_format}"
                title = f"{model_name} — {dataset} ({remasking}) — {metric.upper()} vs. Overhead"
                _write_pareto(
                    perf, lat_data,
                    output_dir / plot_name,
                    metric=metric, title=title,
                )
                n_plots += 1

            if n_plots == 0:
                print("[report] No matching (performance, latency) pairs found for Pareto plots")

    print(f"[report] Done. Output in {output_dir}")


if __name__ == "__main__":
    main()
