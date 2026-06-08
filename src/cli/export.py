"""CLI entry point for XLSX export of UQ evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.io.results_xlsx import (
    aggregate_results,
    aggregate_summary_columns,
    find_kfold_summaries,
    find_metrics_files,
    load_kfold_results,
    load_run_result,
    write_workbook,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export UQ metrics to XLSX workbook.")
    parser.add_argument("--outputs-root", action="append", default=None,
                        help="Output root to scan; repeat for multiple roots. Default: outputs")
    parser.add_argument("--metrics-name", default="uq_eval_metrics.json")
    parser.add_argument("--metrics-json", action="append", default=None,
                        help="Specific metrics JSON file to include")
    parser.add_argument("--output", default="outputs/uq_results.xlsx")
    parser.add_argument("--report-metrics", nargs="+", default=None,
                        help="Metrics to include: auroc, auprc, prr, ece, brier, spearman_rho, kendall_tau")
    parser.add_argument("--empty-ok", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    roots = args.outputs_root or ["outputs"]

    kfold_files = find_kfold_summaries(roots)
    kfold_run_dirs: set[Path] = set()
    results: list = []
    for kf in kfold_files:
        kfold_run_dirs.add(kf.parent.parent.resolve())
        results.extend(load_kfold_results(kf))

    discovered = find_metrics_files(roots, metrics_name=args.metrics_name)
    discovered = [
        f for f in discovered
        if not any(f.resolve().is_relative_to(d) for d in kfold_run_dirs)
    ]
    explicit = [Path(p).expanduser().resolve() for p in (args.metrics_json or [])]

    metric_files = []
    seen = set()
    for path in [*explicit, *discovered]:
        if path.exists() and path not in seen:
            metric_files.append(path)
            seen.add(path)

    if not metric_files and not results and not args.empty_ok:
        raise SystemExit(f"No {args.metrics_name} files found under: {', '.join(str(r) for r in roots)}")

    results.extend(load_run_result(path) for path in metric_files)
    summaries = aggregate_results(results)
    write_workbook(summaries, args.output, report_metrics=args.report_metrics)

    num_competitors = sum(len(items) for items in summaries.values())
    num_summary = len(aggregate_summary_columns(summaries))
    print(f"Scanned {len(metric_files)} run(s) across {len(summaries)} dataset(s).")
    print(f"Wrote {num_competitors} competitors and {num_summary} summary columns to {args.output}")


if __name__ == "__main__":
    main()
