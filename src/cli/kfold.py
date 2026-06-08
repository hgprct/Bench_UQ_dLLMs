"""CLI entry point for K-fold cross-validation.

Runs per fold: grid search (step selection) + evaluate baseline + selection
features on test. Aggregates mean/std across folds at the end.

Baseline features must be pre-computed via `python -m src.cli.baseline`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-fold cross-validation over step selection + UQ evaluation.")
    parser.add_argument("--run_dir", required=True, help="Generation run directory (with examples.jsonl, traces.npz, config.json)")
    parser.add_argument("--baseline_features_path", required=True,
                        help="Pre-computed baseline_features.jsonl (from src.cli.baseline)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: <run_dir>/kfold)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_train", type=int, required=True, help="Absolute number of training prompts per fold")
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
    """Load and filter prompts, returning (prompts, T)."""
    from src.mmd.load import load_generated_run
    run = load_generated_run(run_dir, step_view="x0")
    prompts = list(run.prompts)
    if not prompts:
        raise RuntimeError(f"No prompts found in {run_dir}")
    T = Counter(p.num_steps for p in prompts).most_common(1)[0][0]
    prompts = [p for p in prompts if p.num_steps == T]
    return prompts, T


def _grid_search_args_for_fold(
    args: argparse.Namespace,
    fold_dir: Path,
    n_train: int,
    n_val: int,
    n_test: int,
) -> argparse.Namespace:
    """Build a grid_search-compatible Namespace for one fold."""
    return SimpleNamespace(
        run_dir=args.run_dir,
        output_dir=str(fold_dir / "selection"),
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


def kfold(args: argparse.Namespace) -> dict[str, Any]:
    from src.datasets.labeling import unlabeled_prompt_ids
    from src.evaluate.runner import evaluate_run
    from src.io.json_utils import read_jsonl, write_jsonl
    from src.kfold import aggregate_kfold_results, write_kfold_summary
    from src.selection.grid_search import grid_search
    from src.split import build_kfold_splits, save_kfold_splits

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "kfold"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts, T = _load_prompts(run_dir)

    records = read_jsonl(run_dir / "examples.jsonl")
    excluded = unlabeled_prompt_ids(records)
    if excluded:
        prompts = [p for p in prompts if p.prompt_id not in excluded]
        print(f"[kfold] Excluded {len(excluded)} prompt(s) with ambiguous labels, {len(prompts)} remain")
        if not prompts:
            raise RuntimeError("No prompts left after excluding ambiguous labels")

    all_prompt_ids = [p.prompt_id for p in prompts]
    pid_to_idx = {p.prompt_id: i for i, p in enumerate(prompts)}

    label_lookup: dict[str, bool] = {}
    for r in records:
        pid = str(r.get("qa_example_id", ""))
        if pid and pid not in excluded:
            ic = r.get("final", {}).get("is_correct")
            if ic is not None and pid not in label_lookup:
                label_lookup[pid] = bool(ic)

    n_pos = sum(1 for pid in all_prompt_ids if label_lookup.get(pid, True))
    n_neg = len(all_prompt_ids) - n_pos
    print(f"[kfold] Class balance: {n_pos} correct, {n_neg} incorrect "
          f"({100.0 * n_neg / max(len(all_prompt_ids), 1):.1f}% minority)")

    n_train = args.n_train
    n = len(all_prompt_ids)
    k = args.n_folds
    if n < 2:
        raise RuntimeError(f"Only {n} prompt(s) after filtering — too few for K-fold (need >= 2)")
    if k > n:
        old_k = k
        k = n
        print(f"[kfold] WARNING: n_folds={old_k} exceeds prompt count ({n}), reducing to {k}")
    max_test_size = n // k + (1 if n % k > 0 else 0)
    max_non_test = n - max_test_size
    if n_train > max_non_test:
        capped = max(1, max_non_test - 1)
        print(f"[kfold] WARNING: n_train={n_train} exceeds non-test budget ({max_non_test}), "
              f"capping to {capped}")
        n_train = capped

    print(f"[kfold] {len(prompts)} prompts, T={T}, {k} folds, n_train={n_train}")

    folds = build_kfold_splits(
        all_prompt_ids, k=k, n_train=n_train, seed=args.seed,
        labels=label_lookup if label_lookup else None,
    )
    save_kfold_splits(
        folds, output_dir / "kfold_splits.json",
        k=k, n_train=n_train, seed=args.seed,
    )

    for i, fold in enumerate(folds):
        print(f"  Fold {i}: train={len(fold['train'])}, val={len(fold['val'])}, test={len(fold['test'])}")

    print(f"[kfold] Loading pre-computed baseline features from {args.baseline_features_path}")
    baseline_features = read_jsonl(args.baseline_features_path)
    if excluded:
        baseline_features = [r for r in baseline_features if str(r["prompt_id"]) not in excluded]

    baseline_by_pid = {str(r["prompt_id"]): r for r in baseline_features}

    fold_results: list[dict[str, Any]] = []
    for fold_idx, fold in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[kfold] Fold {fold_idx}/{k - 1}")
        print(f"{'='*60}")

        train_idx = sorted([pid_to_idx[pid] for pid in fold["train"]])
        val_idx = sorted([pid_to_idx[pid] for pid in fold["val"]])
        test_idx = sorted([pid_to_idx[pid] for pid in fold["test"]])

        gs_args = _grid_search_args_for_fold(
            args, fold_dir,
            n_train=len(train_idx),
            n_val=len(val_idx),
            n_test=len(test_idx),
        )

        print(f"[kfold] Running grid search for fold {fold_idx}...")
        grid_result = grid_search(gs_args, split_override=(train_idx, val_idx, test_idx))

        test_baseline = [baseline_by_pid[pid] for pid in fold["test"] if pid in baseline_by_pid]

        baseline_dir = fold_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_features_path = baseline_dir / "uq_features.jsonl"
        write_jsonl(baseline_features_path, test_baseline)
        print(f"[kfold] Evaluating baseline features on fold {fold_idx} test set ({len(test_baseline)} prompts)...")

        baseline_eval = evaluate_run(
            baseline_features_path,
            output_json=baseline_dir / "uq_eval_metrics.json",
            output_csv=baseline_dir / "uq_eval_metrics.csv",
            bootstrap=True,
            bootstrap_samples=args.bootstrap_samples,
        )

        selection_eval = grid_result.get("test_eval", {})

        fold_results.append({
            "fold": fold_idx,
            "test_size": len(test_idx),
            "best_lambda": grid_result.get("best_lambda"),
            "selected_steps": grid_result.get("selected_steps"),
            "baseline": baseline_eval,
            "selection": selection_eval,
        })

        print(f"[kfold] Fold {fold_idx} done.")

    print(f"\n{'='*60}")
    print(f"[kfold] Aggregating results across {k} folds...")
    print(f"{'='*60}")

    summary = aggregate_kfold_results(fold_results)
    write_kfold_summary(
        summary, fold_results, output_dir,
        meta={
            "run_dir": str(run_dir),
            "n_folds": k,
            "n_train": n_train,
            "n_total_prompts": len(prompts),
            "T": T,
            "seed": args.seed,
            "lambda_values": sorted(set(args.lambda_values)),
            "budget_k": args.budget_k,
        },
    )

    _print_summary(summary)
    print(f"\n[kfold] Results saved to {output_dir}")
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    for section in ("baseline", "selection"):
        features = summary.get(section, {})
        if not features:
            continue
        print(f"\n  {section.upper()} features:")
        for feat_name, metrics in sorted(features.items()):
            parts = []
            for m in ("auroc", "auprc", "prr"):
                vals = metrics.get(m)
                if vals:
                    parts.append(f"{m}={vals['mean']:.4f}+/-{vals['std']:.4f}")
            if parts:
                print(f"    {feat_name}: {', '.join(parts)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    kfold(args)


if __name__ == "__main__":
    main()
