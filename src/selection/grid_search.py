"""Lambda grid search with train / validation / test splits."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.io.json_utils import read_json, read_jsonl, to_jsonable, write_jsonl
from src.mmd.kernels import EmbeddingKernel
from src.mmd.load import is_valid_text, load_generated_run
from src.selection.evaluate import discretize, evaluate_split, prefetch_nli_for_split
from src.selection.train import (
    compute_averaged_kernels,
    compute_delta_U,
    solve_qp,
    three_way_split,
)

_SAMPLING_FEATURE_BASES = ("se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern")


def _best_metric_value(
    uq_metrics: dict[str, dict[str, Any]],
    metric: str,
    feature: str | None = None,
) -> tuple[float | None, str | None]:
    if feature:
        for name in (feature, f"selected-{feature}"):
            val = uq_metrics.get(name, {}).get(metric)
            try:
                return float(val), name
            except (TypeError, ValueError):
                continue
        return None, feature
    best_val = None
    best_feat = None
    for feat_name, feat_data in uq_metrics.items():
        raw = feat_data.get(metric)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if best_val is None or v > best_val:
            best_val = v
            best_feat = feat_name
    return best_val, best_feat


def _mean_metric_over_sampling_features(
    uq_metrics: dict[str, dict[str, Any]],
    metric: str,
) -> float | None:
    values = []
    for base in _SAMPLING_FEATURE_BASES:
        feat_data = uq_metrics.get(f"selected-{base}", {})
        raw = feat_data.get(metric)
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def build_selection_feature_rows(
    prompts,
    indices,
    selected_steps,
    *,
    entailment_model,
    record_lookup,
    use_greedy=True,
    trajectory_index=0,
    nli_batch_size=8,
    w_star=None,
):
    from src.features.sampling import compute_sampling_features

    rows = []
    print(f"  Building feature rows for {len(indices)} prompts...")
    for count, idx in enumerate(indices):
        prompt = prompts[idx]
        if count % 50 == 0:
            print(f"    [{count+1}/{len(indices)}] prompt={prompt.prompt_id}")

        row: dict[str, Any] = {"prompt_id": prompt.prompt_id}

        samples = sorted(prompt.samples, key=lambda s: s.response_sample_id)
        target = None
        if samples:
            if use_greedy:
                target = samples[0]
            else:
                sampled = [s for s in samples if s.response_sample_id > 0]
                pool = sampled or samples
                target = pool[trajectory_index] if trajectory_index < len(pool) else None

        prompt_records = record_lookup.get(prompt.prompt_id, [])
        is_correct = None
        if target is not None:
            for rec in prompt_records:
                rsid = rec.get("response_sample_id")
                if rsid is None:
                    meta = rec.get("metadata")
                    if isinstance(meta, dict):
                        rsid = meta.get("response_sample_id")
                if rsid is not None and int(rsid) == target.response_sample_id:
                    final = rec.get("final", {})
                    is_correct = final.get("is_correct", rec.get("is_correct"))
                    break
            if is_correct is None:
                for rec in prompt_records:
                    final = rec.get("final", {})
                    ic = final.get("is_correct", rec.get("is_correct"))
                    if ic is not None:
                        is_correct = ic
                        break
        row["is_correct"] = is_correct

        for feat in _SAMPLING_FEATURE_BASES:
            row[feat] = None
        row["ad"] = None

        if target is not None and entailment_model is not None:
            texts = []
            for step_idx in selected_steps:
                si = int(step_idx)
                if si < len(target.step_answers) and is_valid_text(target.step_answers[si]):
                    texts.append(target.step_answers[si].strip())

            if texts:
                features = compute_sampling_features(
                    texts=texts,
                    nli_model=entailment_model if len(texts) > 1 else None,
                    nli_batch_size=nli_batch_size,
                )
                for k, v in features.items():
                    row[k] = v

            if w_star is not None:
                from src.features.trajectory import compute_averaged_dissimilarity
                row["ad"] = compute_averaged_dissimilarity(
                    target.step_answers, target.final_answer,
                    entailment_model, weights=w_star,
                    nli_batch_size=nli_batch_size,
                )

        rows.append(row)

    return rows


def grid_search(
    args: argparse.Namespace,
    *,
    split_override: tuple[list[int], list[int], list[int]] | None = None,
) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lambda_values = sorted(set(args.lambda_values))

    print(f"[grid] Loading run from {run_dir}")
    run = load_generated_run(run_dir, step_view="x0")
    prompts = list(run.prompts)
    if not prompts:
        raise RuntimeError(f"No prompts found in {run_dir}")

    T = Counter(p.num_steps for p in prompts).most_common(1)[0][0]
    prompts = [p for p in prompts if p.num_steps == T]
    print(f"[grid] {len(prompts)} prompts with T={T}")

    if split_override is not None:
        train_idx, val_idx, test_idx = split_override
    else:
        train_idx, val_idx, test_idx = three_way_split(
            len(prompts), args.n_train, args.n_val, args.n_test, args.split_seed,
        )
    print(f"[grid] Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    kernel = EmbeddingKernel(
        name=f"embedding-{args.kernel}",
        embedding_model=args.embedding_model,
        rbf_bandwidth=args.rbf_bandwidth,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        normalize_embeddings=True,
    )
    embedder = kernel.embedder

    print(f"[grid] Computing kernel matrices on {len(train_idx)} training prompts...")
    t0 = time.time()
    A_V_bar, A_U_bar, C_bar, n_valid_train = compute_averaged_kernels(
        prompts, train_idx, embedder, kernel,
    )
    kernel_time = time.time() - t0
    print(f"[grid] Kernel matrices done in {kernel_time:.1f}s ({n_valid_train} valid prompts)")

    Delta_U_bar = compute_delta_U(C_bar, A_U_bar)

    A_bar = A_U_bar if args.estimator_type == "unbiased" else A_V_bar
    b_bar = A_bar[:, -1]
    c_bar = A_bar[-1, -1]

    examples_path = run_dir / "examples.jsonl"
    config_path = run_dir / "config.json"
    records = read_jsonl(examples_path) if examples_path.exists() else []
    config = read_json(config_path) if config_path.exists() else {}
    record_lookup: dict[str, list[dict[str, Any]]] = {}
    for idx_r, rec in enumerate(records):
        pid = str(rec.get("qa_example_id") or rec.get("qa_index") or idx_r)
        record_lookup.setdefault(pid, []).append(rec)

    entailment_model = None
    if not args.skip_nli:
        from src.semantic.entailment import load_entailment_model
        from src.features.sampling import CachedEntailmentModel
        print("[grid] Loading NLI model...")
        base_model = load_entailment_model(args.nli_model, device=args.nli_device)
        entailment_model = CachedEntailmentModel(base_model)

    grid_results: list[dict[str, Any]] = []
    selection_metric = args.selection_metric
    anchor_feature = getattr(args, 'anchor_feature', None)
    if anchor_feature:
        selection_mode = "single"
        selection_feature = anchor_feature
        val_feature_filter: list[str] | None = [f"selected-{anchor_feature}"]
    else:
        selection_mode = args.selection_mode
        selection_feature = args.selection_feature
        val_feature_filter = None

    # --- Phase 1: Solve all QPs upfront ---
    print(f"\n[grid] Phase 1: Solving QPs for {len(lambda_values)} lambda values...")
    qp_solutions: list[dict[str, Any]] = []
    for lam in lambda_values:
        print(f"\n  Lambda = {lam}")
        try:
            w_star, qp_diag = solve_qp(
                A_bar, b_bar, Delta_U_bar, c_bar,
                lambda_reg=lam, budget_k=args.budget_k,
            )
        except RuntimeError as e:
            print(f"  QP failed: {e}")
            qp_solutions.append({"lambda_reg": lam, "status": "qp_failed", "error": str(e)})
            continue

        selected_steps = discretize(w_star, args.budget_k, args.discretization_method)
        k = len(selected_steps)
        print(f"  Selected {k} steps: {selected_steps.tolist()}")

        lam_dir = output_dir / f"lambda_{lam}"
        lam_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            lam_dir / "trained_weights.npz",
            w_star=w_star,
            A_V_bar=A_V_bar, A_U_bar=A_U_bar, C_bar=C_bar,
            Delta_U_bar=Delta_U_bar, b_bar=b_bar,
            train_indices=np.array(train_idx),
            val_indices=np.array(val_idx),
            test_indices=np.array(test_idx),
        )

        qp_solutions.append({
            "lambda_reg": lam,
            "status": "ok",
            "w_star": w_star,
            "selected_steps": selected_steps,
        })

    # --- Phase 2 & 3: Val evaluation (skip when single lambda + no val) ---
    valid_solutions = [s for s in qp_solutions if s["status"] == "ok"]
    skip_val = len(valid_solutions) == 1 and not val_idx

    if skip_val:
        sol = valid_solutions[0]
        grid_results.append({
            "lambda_reg": sol["lambda_reg"],
            "status": "ok",
            "selected_steps": sol["selected_steps"].tolist(),
            "w_star_nnz": int(np.sum(sol["w_star"] > 1e-8)),
            "val_selection_metric_value": 0.0,
            "val_selection_feature": None,
        })
        print(f"[grid] Single lambda with no val split — skipping validation")
    else:
        if valid_solutions and not args.skip_nli and entailment_model is not None:
            from src.features.sampling import CachedEntailmentModel
            if isinstance(entailment_model, CachedEntailmentModel):
                union_steps = np.array(sorted(set().union(
                    *(set(s["selected_steps"].tolist()) for s in valid_solutions)
                )))
                n_pairs = prefetch_nli_for_split(
                    prompts, val_idx, union_steps, entailment_model,
                    use_greedy=args.use_greedy,
                    trajectory_index=args.trajectory_index,
                    nli_batch_size=args.nli_batch_size,
                )
                if n_pairs:
                    print(f"[grid] Pre-warmed NLI cache for val: {n_pairs} pairs "
                          f"({len(union_steps)} unique steps from {len(valid_solutions)} lambdas)")

        embedding_cache: dict[int, tuple] = {}
        for sol in qp_solutions:
            lam = sol["lambda_reg"]
            if sol["status"] != "ok":
                grid_results.append(sol)
                continue

            print(f"\n{'=' * 60}\n[grid] Evaluating lambda = {lam} on val\n{'=' * 60}")

            val_result = evaluate_split(
                prompts, val_idx, sol["selected_steps"], len(sol["selected_steps"]),
                kernel=kernel, embedder=embedder,
                entailment_model=entailment_model,
                record_lookup=record_lookup,
                split_name="val",
                nli_batch_size=args.nli_batch_size,
                use_greedy=args.use_greedy,
                trajectory_index=args.trajectory_index,
                skip_nli=args.skip_nli,
                full_traj=args.full_traj,
                w_star=sol["w_star"],
                discretization_method=args.discretization_method,
                feature_filter=val_feature_filter,
                embedding_cache=embedding_cache,
            )

            uq_metrics = val_result.get("uq_metrics", {})
            if selection_mode == "all_sampling":
                sel_val = _mean_metric_over_sampling_features(uq_metrics, selection_metric)
                sel_feat = "all_sampling"
            else:
                sel_val, sel_feat = _best_metric_value(uq_metrics, selection_metric, selection_feature)

            grid_results.append({
                "lambda_reg": lam,
                "status": "ok",
                "selected_steps": sol["selected_steps"].tolist(),
                "w_star_nnz": int(np.sum(sol["w_star"] > 1e-8)),
                "val_selection_metric_value": sel_val,
                "val_selection_feature": sel_feat,
                "val_mmd2_plugin_mean": val_result.get("distributional", {}).get("mmd2_plugin_mean"),
                "val_redundancy_mean": val_result.get("redundancy", {}).get("mean"),
            })

    valid_results = [
        r for r in grid_results
        if r["status"] == "ok" and r.get("val_selection_metric_value") is not None
    ]

    if not valid_results:
        print("\n[grid] No valid lambda produced a measurable validation metric.")
        summary: dict[str, Any] = {
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "lambda_values": lambda_values,
            "selection_mode": selection_mode,
            "grid_results": grid_results,
            "best_lambda": None,
        }
        with (output_dir / "grid_search_result.json").open("w") as f:
            json.dump(to_jsonable(summary), f, indent=2, allow_nan=True)
        return summary

    best_entry = max(valid_results, key=lambda r: r["val_selection_metric_value"])
    best_lambda = best_entry["lambda_reg"]

    print(f"\n{'=' * 60}")
    print(f"[grid] Best lambda = {best_lambda} "
          f"(val {selection_metric.upper()} = {best_entry['val_selection_metric_value']:.4f}, "
          f"mode={selection_mode})")
    print(f"{'=' * 60}")

    best_lam_dir = output_dir / f"lambda_{best_lambda}"
    shutil.copy2(best_lam_dir / "trained_weights.npz", output_dir / "trained_weights.npz")
    w_star_best = np.load(best_lam_dir / "trained_weights.npz")["w_star"]
    selected_steps_best = discretize(w_star_best, args.budget_k, args.discretization_method)
    k_best = len(selected_steps_best)

    print(f"\n[grid] Building test features ({len(test_idx)} prompts) with selected steps: {selected_steps_best.tolist()}")
    if not args.skip_nli and entailment_model is not None:
        from src.features.sampling import CachedEntailmentModel
        if isinstance(entailment_model, CachedEntailmentModel):
            n_pairs = prefetch_nli_for_split(
                prompts, test_idx, selected_steps_best, entailment_model,
                use_greedy=args.use_greedy,
                trajectory_index=args.trajectory_index,
                nli_batch_size=args.nli_batch_size,
            )
            if n_pairs:
                print(f"[grid] Warmed NLI cache for test ({n_pairs} pairs)")
    test_feature_rows = build_selection_feature_rows(
        prompts, test_idx, selected_steps_best,
        entailment_model=entailment_model,
        record_lookup=record_lookup,
        use_greedy=args.use_greedy,
        trajectory_index=args.trajectory_index,
        nli_batch_size=args.nli_batch_size,
        w_star=w_star_best,
    )

    features_path = output_dir / "uq_features.jsonl"
    write_jsonl(features_path, test_feature_rows)
    print(f"[grid] Wrote {len(test_feature_rows)} feature rows to {features_path}")

    from src.evaluate.runner import evaluate_run
    metrics_json = output_dir / "uq_eval_metrics.json"
    metrics_csv = output_dir / "uq_eval_metrics.csv"
    print("[grid] Evaluating test features...")
    test_eval = evaluate_run(
        features_path,
        output_json=metrics_json,
        output_csv=metrics_csv,
        bootstrap=True,
        bootstrap_samples=1000,
    )

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "T": T,
        "budget_k": args.budget_k,
        "estimator_type": args.estimator_type,
        "selection_mode": selection_mode,
        "selection_metric": selection_metric,
        "selection_feature": selection_feature,
        "split_seed": args.split_seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "lambda_values": lambda_values,
        "grid_results": grid_results,
        "best_lambda": best_lambda,
        "best_val_metric_value": best_entry["val_selection_metric_value"],
        "selected_steps": selected_steps_best.tolist(),
        "test_eval": test_eval,
    }
    with (output_dir / "grid_search_result.json").open("w") as f:
        json.dump(to_jsonable(summary), f, indent=2, allow_nan=True)

    print(f"\n[grid] All results saved to {output_dir}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lambda grid search with train/val/test splits")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_values", type=float, nargs="+", required=True)

    g = parser.add_argument_group("Data split")
    g.add_argument("--n_train", type=int, required=True)
    g.add_argument("--n_val", type=int, required=True)
    g.add_argument("--n_test", type=int, required=True)
    g.add_argument("--split_seed", type=int, default=42)

    g = parser.add_argument_group("QP parameters")
    g.add_argument("--budget_k", type=int, default=8)
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
    g.add_argument("--full_traj", action="store_true",
                   help="Also compute full-trajectory baseline features (full-*) alongside selected-*")
    g.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    g.add_argument("--nli_device", default=None)
    g.add_argument("--nli_batch_size", type=int, default=256)

    g = parser.add_argument_group("Best-lambda selection")
    g.add_argument("--selection_mode", default="all_sampling",
                   choices=["all_sampling", "single"],
                   help="'all_sampling': average metric across all sampling features (default); "
                        "'single': optimize a single feature (requires --selection_feature)")
    g.add_argument("--selection_metric", default="auroc",
                   choices=["auroc", "auprc", "prr", "ece", "brier", "spearman_rho", "kendall_tau"])
    g.add_argument("--selection_feature", default=None,
                   help="Feature to optimize in 'single' mode")
    g.add_argument("--anchor_feature", default=None,
                   choices=["se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern"],
                   help="Single sampling feature evaluated on val for lambda selection; "
                        "all features are still built on test. Overrides --selection_mode/--selection_feature.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    grid_search(args)


if __name__ == "__main__":
    main()
