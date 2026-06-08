"""CLI entry point for single-split quick evaluation.

Fast alternative to K-fold: one train/val/test split, one grid search pass.
When a single lambda is provided, val is merged into test (no selection needed).

Baseline features must be pre-computed via `python -m src.cli.baseline`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-split quick evaluation over step selection + UQ evaluation.",
    )
    parser.add_argument("--run_dir", required=True,
                        help="Generation run directory (with examples.jsonl, traces.npz, config.json)")
    parser.add_argument("--baseline_features_path", required=True,
                        help="Pre-computed baseline_features.jsonl (from src.cli.baseline)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <run_dir>/quickeval)")
    parser.add_argument("--n_train", type=int, required=100,
                        help="Absolute number of training prompts")
    parser.add_argument("--n_val", type=int, default=300,
                        help="Absolute number of validation prompts (default: auto from remaining)")
    parser.add_argument("--n_test", type=int, default=None,
                        help="Absolute number of test prompts (default: all remaining)")
    parser.add_argument("--seed", type=int, default=42)

    g = parser.add_argument_group("Grid search / QP")
    g.add_argument("--lambda_values", type=float, nargs="+", required=True)
    g.add_argument("--budget_k", type=int, default=20)
    g.add_argument("--estimator_type", choices=["unbiased", "convex_surrogate"], default="unbiased")

    g = parser.add_argument_group("Kernel / embedding")
    g.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    g.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "cosine"])
    g.add_argument("--rbf_bandwidth", default="median")
    g.add_argument("--embedding_batch_size", type=int, default=64)
    g.add_argument("--embedding_device", default="auto")

    g = parser.add_argument_group("Evaluation")
    g.add_argument("--discretization_method", default="top_k", choices=["top_k", "randomized"])
    g.add_argument("--use_greedy", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--trajectory_index", type=int, default=0)
    g.add_argument("--skip_nli", action="store_true")
    g.add_argument("--full_traj", action="store_true")
    g.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    g.add_argument("--nli_device", default=None)
    g.add_argument("--nli_batch_size", type=int, default=256)
    g.add_argument("--bootstrap_samples", type=int, default=1000)

    g = parser.add_argument_group("Best-lambda selection")
    g.add_argument("--selection_mode", default="all_sampling",
                   choices=["all_sampling", "single"])
    g.add_argument("--selection_metric", default="prr",
                   choices=["auroc", "auprc", "prr", "ece", "brier", "spearman_rho", "kendall_tau"])
    g.add_argument("--selection_feature", default=None)
    g.add_argument("--anchor_feature", default="eigval",
                   choices=["se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern"],
                   help="Single sampling feature evaluated on val for lambda selection; "
                        "all features are still built on test. Overrides --selection_mode/--selection_feature.")

    return parser.parse_args(argv)


def _load_prompts(run_dir: Path) -> tuple[list, int]:
    from src.mmd.load import load_generated_run
    run = load_generated_run(run_dir, step_view="x0")
    prompts = list(run.prompts)
    if not prompts:
        raise RuntimeError(f"No prompts found in {run_dir}")
    T = Counter(p.num_steps for p in prompts).most_common(1)[0][0]
    prompts = [p for p in prompts if p.num_steps == T]
    return prompts, T


def _grid_search_args(
    args: argparse.Namespace,
    output_dir: Path,
    n_train: int,
    n_val: int,
    n_test: int,
) -> argparse.Namespace:
    return SimpleNamespace(
        run_dir=args.run_dir,
        output_dir=str(output_dir / "selection"),
        lambda_values=args.lambda_values,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        split_seed=args.seed,
        budget_k=args.budget_k,
        estimator_type=args.estimator_type,
        embedding_model=args.embedding_model,
        kernel=args.kernel,
        rbf_bandwidth=args.rbf_bandwidth,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        discretization_method=args.discretization_method,
        use_greedy=args.use_greedy,
        trajectory_index=args.trajectory_index,
        skip_nli=args.skip_nli,
        full_traj=args.full_traj,
        nli_model=args.nli_model,
        nli_device=args.nli_device,
        nli_batch_size=args.nli_batch_size,
        selection_mode=args.selection_mode,
        selection_metric=args.selection_metric,
        selection_feature=args.selection_feature,
        anchor_feature=getattr(args, 'anchor_feature', None),
    )


def _copy_config(run_dir: Path, target_dir: Path) -> None:
    config_src = run_dir / "config.json"
    if config_src.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_src, target_dir / "config.json")


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    columns = ["type", "feature", "metric", "value"]
    rows: list[list[Any]] = []
    for section in ("baseline", "selection"):
        features = summary.get(section, {}).get("features", {})
        for feat_name in sorted(features):
            metrics = features[feat_name]
            for metric_name in sorted(metrics):
                val = metrics[metric_name]
                if val is None:
                    continue
                try:
                    float(val)
                except (TypeError, ValueError):
                    continue
                rows.append([section, feat_name, metric_name, f"{float(val):.6f}"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def quickeval(args: argparse.Namespace) -> dict[str, Any]:
    from src.datasets.labeling import unlabeled_prompt_ids
    from src.evaluate.runner import evaluate_run
    from src.io.json_utils import read_jsonl, to_jsonable, write_json, write_jsonl
    from src.selection.grid_search import grid_search
    from src.split import save_splits, split_prompts

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "quickeval"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts, T = _load_prompts(run_dir)

    records = read_jsonl(run_dir / "examples.jsonl")
    excluded = unlabeled_prompt_ids(records)
    if excluded:
        prompts = [p for p in prompts if p.prompt_id not in excluded]
        print(f"[quickeval] Excluded {len(excluded)} prompt(s) with ambiguous labels, {len(prompts)} remain")
        if not prompts:
            raise RuntimeError("No prompts left after excluding ambiguous labels")

    all_prompt_ids = [p.prompt_id for p in prompts]
    pid_to_idx = {p.prompt_id: i for i, p in enumerate(prompts)}
    n = len(all_prompt_ids)

    n_train = args.n_train
    lambda_values = sorted(set(args.lambda_values))
    single_lambda = len(lambda_values) == 1

    if single_lambda:
        n_val = 0
        n_test = n - n_train
        print(f"[quickeval] Single lambda ({lambda_values[0]}): merging val into test")
    else:
        n_val = args.n_val
        n_test = args.n_test
        if n_val is None and n_test is None:
            remaining = n - n_train
            n_val = remaining // 3
            n_test = remaining - n_val
        elif n_val is None:
            n_val = n - n_train - n_test
        elif n_test is None:
            n_test = n - n_train - n_val

    if n_train + n_val + n_test > n:
        raise RuntimeError(
            f"Requested {n_train}+{n_val}+{n_test}={n_train + n_val + n_test} prompts "
            f"but only {n} available"
        )

    print(f"[quickeval] {n} prompts, T={T}, split: train={n_train} val={n_val} test={n_test}")

    splits = split_prompts(all_prompt_ids, train=n_train, val=n_val, test=n_test, seed=args.seed)
    save_splits(splits, output_dir / "splits.json",
                seed=args.seed, train=n_train, val=n_val, test=n_test)

    train_idx = sorted([pid_to_idx[pid] for pid in splits["train"]])
    val_idx = sorted([pid_to_idx[pid] for pid in splits["val"]]) if splits["val"] else []
    test_idx = sorted([pid_to_idx[pid] for pid in splits["test"]])

    gs_args = _grid_search_args(args, output_dir, n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx))
    print(f"[quickeval] Running grid search...")
    grid_result = grid_search(gs_args, split_override=(train_idx, val_idx, test_idx))

    _copy_config(run_dir, output_dir / "selection")

    print(f"[quickeval] Loading pre-computed baseline features from {args.baseline_features_path}")
    baseline_features = read_jsonl(args.baseline_features_path)
    if excluded:
        baseline_features = [r for r in baseline_features if str(r["prompt_id"]) not in excluded]

    baseline_by_pid = {str(r["prompt_id"]): r for r in baseline_features}
    test_baseline = [baseline_by_pid[pid] for pid in splits["test"] if pid in baseline_by_pid]

    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_features_path = baseline_dir / "uq_features.jsonl"
    write_jsonl(baseline_features_path, test_baseline)
    print(f"[quickeval] Evaluating baseline features on test set ({len(test_baseline)} prompts)...")

    baseline_eval = evaluate_run(
        baseline_features_path,
        output_json=baseline_dir / "uq_eval_metrics.json",
        output_csv=baseline_dir / "uq_eval_metrics.csv",
        bootstrap=True,
        bootstrap_samples=args.bootstrap_samples,
    )

    _copy_config(run_dir, baseline_dir)

    selection_eval = grid_result.get("test_eval", {})

    summary = {
        "run_dir": str(run_dir),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_total_prompts": n,
        "T": T,
        "seed": args.seed,
        "lambda_values": lambda_values,
        "budget_k": args.budget_k,
        "single_lambda": single_lambda,
        "best_lambda": grid_result.get("best_lambda"),
        "selected_steps": grid_result.get("selected_steps"),
        "baseline": baseline_eval,
        "selection": selection_eval,
    }

    write_json(output_dir / "quickeval_summary.json", to_jsonable(summary))
    _write_summary_csv(output_dir / "quickeval_summary.csv", summary)

    _print_summary(summary)
    print(f"\n[quickeval] Results saved to {output_dir}")
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    for section in ("baseline", "selection"):
        features = summary.get(section, {}).get("features", {})
        if not features:
            continue
        print(f"\n  {section.upper()} features:")
        for feat_name in sorted(features):
            metrics = features[feat_name]
            parts = []
            for m in ("auroc", "auprc", "prr"):
                val = metrics.get(m)
                if val is not None:
                    try:
                        parts.append(f"{m}={float(val):.4f}")
                    except (TypeError, ValueError):
                        pass
            if parts:
                print(f"    {feat_name}: {', '.join(parts)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    quickeval(args)


if __name__ == "__main__":
    main()
