"""Evaluate learned step selection: distributional MMD and downstream UQ metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.io.json_utils import to_jsonable
from src.mmd.kernels import EmbeddingKernel
from src.mmd.load import (
    PromptTrajectories,
    TrajectorySample,
    is_valid_text,
    load_generated_run,
)
from src.selection.train import compute_kernel_matrices, step_embeddings, three_way_split

BASELINE_METHODS = {"uniform", "random", "last_k"}
LEARNED_METHODS = {"top_k", "randomized"}
NLI_FEATURES = {"ecc", "eigval", "kle-heat", "kle-matern"}


def load_weights(weights_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = dict(np.load(weights_path, allow_pickle=False))
    w_star = data.pop("w_star")
    return w_star, data


def top_k_selection(w_star: np.ndarray, k: int) -> np.ndarray:
    indices = np.argsort(w_star)[::-1][:k]
    return np.sort(indices)


def randomized_rounding(w_star: np.ndarray, k: int, *, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = np.clip(w_star, 0, None)
    total = probs.sum()
    if total < 1e-12:
        return np.sort(rng.choice(len(w_star), size=k, replace=False))
    probs = probs / total
    indices = rng.choice(len(w_star), size=k, replace=False, p=probs)
    return np.sort(indices)


def uniform_selection(T: int, k: int) -> np.ndarray:
    if k >= T:
        return np.arange(T)
    indices = np.round(np.linspace(0, T - 1, k)).astype(int)
    return np.unique(indices)


def last_k_selection(T: int, k: int) -> np.ndarray:
    k = min(k, T)
    return np.arange(T - k, T)


def random_selection(T: int, k: int, *, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, T)
    return np.sort(rng.choice(T, size=k, replace=False))


def discretize(
    w_star: np.ndarray,
    k: int,
    method: str,
    *,
    seed: int | None = None,
) -> np.ndarray:
    T = len(w_star)
    if method == "uniform":
        return uniform_selection(T, k)
    elif method == "last_k":
        return last_k_selection(T, k)
    elif method == "random":
        return random_selection(T, k, seed=seed)

    effective_k = int(np.sum(w_star > 1e-12))
    k = max(min(k, effective_k), 1)

    if method == "top_k":
        return top_k_selection(w_star, k)
    elif method == "randomized":
        return randomized_rounding(w_star, k, seed=seed)
    else:
        raise ValueError(f"Unknown discretization method: {method}")


def binarized_weight_vector(selected_steps: np.ndarray, T: int, k: int) -> np.ndarray:
    w_S = np.zeros(T, dtype=np.float64)
    w_S[selected_steps] = 1.0 / k
    return w_S


def _mmd2_plugin(w_S: np.ndarray, A_V: np.ndarray) -> float:
    self_energy = float(w_S @ A_V @ w_S)
    cross_term = float(A_V[:, -1] @ w_S)
    ref_energy = float(A_V[-1, -1])
    return self_energy - 2.0 * cross_term + ref_energy


def _mmd2_unbiased(w_S: np.ndarray, A_U: np.ndarray) -> float:
    self_energy = float(w_S @ A_U @ w_S)
    cross_term = float(A_U[:, -1] @ w_S)
    ref_energy = float(A_U[-1, -1])
    return self_energy - 2.0 * cross_term + ref_energy


def _prompt_distributional_metrics(
    emb: np.ndarray,
    kernel: EmbeddingKernel,
    selected_steps: np.ndarray,
    k: int,
) -> dict[str, float]:
    N, T, D = emb.shape
    if N < 2 or T == 0:
        return {"mmd2_plugin": float("nan"), "mmd2_unbiased": float("nan"), "redundancy": float("nan"), "penalty": float("nan")}

    A_V, A_U, C = compute_kernel_matrices(emb, kernel)
    w_S = binarized_weight_vector(selected_steps, T, k)

    return {
        "mmd2_plugin": _mmd2_plugin(w_S, A_V),
        "mmd2_unbiased": _mmd2_unbiased(w_S, A_U),
        "redundancy": float(w_S @ C @ w_S),
        "penalty": float(w_S @ (C - A_U) @ w_S),
    }


def _select_trajectory(
    prompt: PromptTrajectories,
    *,
    use_greedy: bool = True,
    trajectory_index: int = 0,
) -> TrajectorySample | None:
    samples = sorted(prompt.samples, key=lambda s: s.response_sample_id)
    if not samples:
        return None
    if use_greedy:
        return samples[0]
    sampled = [s for s in samples if s.response_sample_id > 0]
    if not sampled:
        return samples[trajectory_index] if trajectory_index < len(samples) else None
    return sampled[trajectory_index] if trajectory_index < len(sampled) else None


_UQ_FEATURE_BASES = ("se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern")
_TRAJECTORY_ONLY_BASES = ("ad",)


def compute_trajectory_features(
    sample: TrajectorySample,
    selected_steps: np.ndarray,
    entailment_model: Any,
    *,
    prefix: str,
    nli_batch_size: int = 8,
) -> dict[str, float | None]:
    from src.features.sampling import compute_sampling_features

    texts: list[str] = []
    for step_idx in selected_steps:
        step_idx = int(step_idx)
        if step_idx < len(sample.step_answers) and is_valid_text(sample.step_answers[step_idx]):
            texts.append(sample.step_answers[step_idx].strip())

    if not texts:
        return {f"{prefix}-{k}": None for k in _UQ_FEATURE_BASES}

    features = compute_sampling_features(
        texts=texts,
        nli_model=entailment_model if len(texts) > 1 else None,
        nli_batch_size=nli_batch_size,
    )
    return {f"{prefix}-{k}": v for k, v in features.items()}


def _record_for_response_sample_id(
    prompt_records: list[dict[str, Any]],
    target_rsid: int,
) -> dict[str, Any] | None:
    for rec in prompt_records:
        rsid = rec.get("response_sample_id")
        if rsid is None:
            meta = rec.get("metadata")
            if isinstance(meta, dict):
                rsid = meta.get("response_sample_id")
        if rsid is not None and int(rsid) == target_rsid:
            return rec
    return None


def _error_label_from_row(record: dict[str, Any]) -> int | None:
    final = record.get("final", {})
    is_correct = final.get("is_correct")
    if is_correct is None:
        is_correct = record.get("is_correct")
    if is_correct is None:
        return None
    return 0 if bool(is_correct) else 1


def prefetch_nli_for_split(
    prompts: list,
    indices: list[int],
    selected_steps: np.ndarray,
    entailment_model: Any,
    *,
    use_greedy: bool = True,
    trajectory_index: int = 0,
    nli_batch_size: int = 8,
    include_ad: bool = True,
) -> int:
    """Batch-warm the NLI cache for all prompts in a split in one GPU pass.

    Returns total (premise, hypothesis) pairs submitted. Only effective when
    entailment_model is a CachedEntailmentModel (cache hits on per-prompt calls).
    """
    from src.mmd.load import is_valid_text

    all_premises: list[str] = []
    all_hypotheses: list[str] = []
    for idx in indices:
        prompt = prompts[idx]
        target = _select_trajectory(prompt, use_greedy=use_greedy, trajectory_index=trajectory_index)
        if target is None:
            continue

        texts = list(dict.fromkeys(
            target.step_answers[int(s)].strip()
            for s in selected_steps
            if int(s) < len(target.step_answers) and is_valid_text(target.step_answers[int(s)])
        ))
        if len(texts) >= 2:
            for left in texts:
                for right in texts:
                    all_premises.append(left)
                    all_hypotheses.append(right)

        if include_ad and is_valid_text(target.final_answer):
            final_text = target.final_answer.strip()
            seen: set[str] = set()
            for t in range(len(target.step_answers)):
                text = target.step_answers[t]
                if not is_valid_text(text):
                    continue
                text = text.strip()
                if text == final_text or text in seen:
                    continue
                seen.add(text)
                all_premises.extend([text, final_text])
                all_hypotheses.extend([final_text, text])

    if all_premises:
        entailment_model.batch_probabilities(all_premises, all_hypotheses, batch_size=nli_batch_size)
    return len(all_premises)


def evaluate_split(
    prompts: list[PromptTrajectories],
    indices: list[int],
    selected_steps: np.ndarray,
    k: int,
    *,
    kernel: EmbeddingKernel,
    embedder,
    entailment_model: Any,
    record_lookup: dict[str, list[dict[str, Any]]],
    split_name: str,
    nli_batch_size: int = 8,
    use_greedy: bool = True,
    trajectory_index: int = 0,
    skip_nli: bool = False,
    full_traj: bool = False,
    w_star: np.ndarray | None = None,
    discretization_method: str = "top_k",
    discretize_seed: int = 0,
    feature_filter: list[str] | None = None,
    embedding_cache: dict[int, tuple] | None = None,
) -> dict[str, Any]:
    from src.evaluate.metrics import evaluate_score, orient_scores
    from src.features.directions import FEATURE_DIRECTIONS

    feat_directions = dict(FEATURE_DIRECTIONS)
    for prefix in ("selected", "full"):
        for base, direction in FEATURE_DIRECTIONS.items():
            feat_directions[f"{prefix}-{base}"] = direction

    mmd2_values: list[float] = []
    mmd2_unbiased_values: list[float] = []
    redundancy_values: list[float] = []
    penalty_values: list[float] = []
    uq_rows: list[dict[str, Any]] = []

    per_prompt_randomized = discretization_method == "randomized" and w_star is not None

    if not skip_nli and not per_prompt_randomized and entailment_model is not None:
        from src.features.sampling import CachedEntailmentModel
        if isinstance(entailment_model, CachedEntailmentModel):
            n_pairs = prefetch_nli_for_split(
                prompts, indices, selected_steps, entailment_model,
                use_greedy=use_greedy, trajectory_index=trajectory_index,
                nli_batch_size=nli_batch_size,
            )
            if n_pairs:
                print(f"  [prefetch] {n_pairs} NLI pairs batched across {len(indices)} {split_name} prompts")

    print(f"\n[evaluate] Processing {split_name} split ({len(indices)} prompts)...")

    for count, idx in enumerate(indices):
        prompt = prompts[idx]
        if count % 50 == 0:
            print(f"  [{count+1}/{len(indices)}] prompt={prompt.prompt_id}")

        if per_prompt_randomized:
            prompt_seed = discretize_seed * 100_000 + idx
            prompt_steps = discretize(w_star, k, "randomized", seed=prompt_seed)
        else:
            prompt_steps = selected_steps

        effective_k = len(prompt_steps)
        if embedding_cache is not None and idx in embedding_cache:
            emb, N, _T_prompt = embedding_cache[idx]
        else:
            emb, N, _T_prompt = step_embeddings(prompt, embedder)
            if embedding_cache is not None:
                embedding_cache[idx] = (emb, N, _T_prompt)
        if emb.size > 0 and N >= 2:
            dist_metrics = _prompt_distributional_metrics(emb, kernel, prompt_steps, effective_k)
            mmd2_values.append(dist_metrics["mmd2_plugin"])
            mmd2_unbiased_values.append(dist_metrics["mmd2_unbiased"])
            redundancy_values.append(dist_metrics["redundancy"])
            penalty_values.append(dist_metrics["penalty"])

        target_sample = _select_trajectory(prompt, use_greedy=use_greedy, trajectory_index=trajectory_index)

        uq_features: dict[str, float | None] = {}
        if target_sample is not None and not skip_nli:
            uq_features.update(compute_trajectory_features(
                target_sample, prompt_steps, entailment_model,
                prefix="selected", nli_batch_size=nli_batch_size,
            ))
            from src.features.trajectory import compute_averaged_dissimilarity
            if w_star is not None:
                uq_features["selected-ad"] = compute_averaged_dissimilarity(
                    target_sample.step_answers, target_sample.final_answer,
                    entailment_model, weights=w_star,
                    nli_batch_size=nli_batch_size,
                )
            if full_traj:
                all_steps = np.arange(len(target_sample.step_answers))
                uq_features.update(compute_trajectory_features(
                    target_sample, all_steps, entailment_model,
                    prefix="full", nli_batch_size=nli_batch_size,
                ))
                uq_features["full-ad"] = compute_averaged_dissimilarity(
                    target_sample.step_answers, target_sample.final_answer,
                    entailment_model, nli_batch_size=nli_batch_size,
                )

        prompt_records = record_lookup.get(prompt.prompt_id, [])
        label = None
        if target_sample is not None:
            target_rec = _record_for_response_sample_id(prompt_records, target_sample.response_sample_id)
            if target_rec is not None:
                label = _error_label_from_row(target_rec)
            else:
                for rec in prompt_records:
                    label = _error_label_from_row(rec)
                    if label is not None:
                        break

        uq_rows.append({
            "prompt_id": prompt.prompt_id,
            "features": uq_features,
            "error_label": label,
        })

    mmd2_arr = np.array([v for v in mmd2_values if np.isfinite(v)])
    mmd2_ub_arr = np.array([v for v in mmd2_unbiased_values if np.isfinite(v)])
    red_arr = np.array([v for v in redundancy_values if np.isfinite(v)])
    pen_arr = np.array([v for v in penalty_values if np.isfinite(v)])

    uq_eval_results = {}
    feature_names = [f"selected-{b}" for b in _UQ_FEATURE_BASES]
    if w_star is not None:
        feature_names += [f"selected-{b}" for b in _TRAJECTORY_ONLY_BASES]
    if full_traj:
        feature_names += [f"full-{b}" for b in _UQ_FEATURE_BASES]
        feature_names += [f"full-{b}" for b in _TRAJECTORY_ONLY_BASES]
    if feature_filter is not None:
        feature_names = [f for f in feature_names if f in feature_filter]
    for feat_name in feature_names:
        labels_list: list[int] = []
        scores_list: list[float] = []
        for row in uq_rows:
            label = row["error_label"]
            score = row["features"].get(feat_name)
            if label is not None and score is not None:
                labels_list.append(label)
                scores_list.append(float(score))

        if len(labels_list) >= 10 and len(set(labels_list)) == 2:
            scores_list = orient_scores(scores_list, feat_name, feat_directions)
            result = evaluate_score(labels_list, scores_list)
            uq_eval_results[feat_name] = {
                "auroc": result["auroc"],
                "auprc": result["auprc"],
                "prr": result["prr"],
                "ece": result["ece"],
                "brier": result["brier"],
                "spearman_rho": result["spearman_rho"],
                "kendall_tau": result["kendall_tau"],
                "n_examples": len(labels_list),
            }
        else:
            uq_eval_results[feat_name] = {
                "auroc": None, "auprc": None, "prr": None,
                "ece": None, "brier": None, "spearman_rho": None, "kendall_tau": None,
                "n_examples": len(labels_list),
            }

    return {
        "split": split_name,
        "n_prompts": len(indices),
        "distributional": {
            "mmd2_plugin_mean": float(mmd2_arr.mean()) if mmd2_arr.size else None,
            "mmd2_plugin_std": float(mmd2_arr.std()) if mmd2_arr.size else None,
            "mmd2_unbiased_mean": float(mmd2_ub_arr.mean()) if mmd2_ub_arr.size else None,
            "mmd2_unbiased_std": float(mmd2_ub_arr.std()) if mmd2_ub_arr.size else None,
        },
        "redundancy": {
            "mean": float(red_arr.mean()) if red_arr.size else None,
            "std": float(red_arr.std()) if red_arr.size else None,
        },
        "penalty": {
            "mean": float(pen_arr.mean()) if pen_arr.size else None,
            "std": float(pen_arr.std()) if pen_arr.size else None,
        },
        "uq_metrics": uq_eval_results,
    }


def _build_record_lookup(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for idx, rec in enumerate(records):
        qa_idx = rec.get("qa_example_id") or rec.get("qa_index")
        if qa_idx is not None:
            pid = str(qa_idx)
        else:
            num_samples = int(config.get("num_response_samples", 1) or 1)
            pid = str(idx // num_samples) if num_samples > 1 else str(idx)
        lookup.setdefault(pid, []).append(rec)
    return lookup


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    disc_method = args.discretization_method
    disc_seed = args.seed
    is_baseline = disc_method in BASELINE_METHODS

    if is_baseline:
        weights_dir = Path(args.weights_dir) if args.weights_dir else None
        train_result = {}
        w_star = None

        if weights_dir and (weights_dir / "train_result.json").exists():
            with (weights_dir / "train_result.json").open() as f:
                train_result = json.load(f)
            w_star_loaded, extra = load_weights(weights_dir / "trained_weights.npz")
            train_indices = extra["train_indices"].tolist()
            val_indices = extra.get("val_indices", np.array([])).tolist()
            test_indices = extra["test_indices"].tolist()
            T = len(w_star_loaded)
        else:
            weights_dir = None
            run_cached = load_generated_run(run_dir, step_view="x0")
            prompts_pre = list(run_cached.prompts)
            T = Counter(p.num_steps for p in prompts_pre).most_common(1)[0][0]
            n_filtered = len([p for p in prompts_pre if p.num_steps == T])
            if args.n_train is None or args.n_val is None or args.n_test is None:
                raise SystemExit(
                    "--n_train, --n_val, --n_test are required when running "
                    "baselines without a --weights_dir"
                )
            train_indices, val_indices, test_indices = three_way_split(
                n_filtered, args.n_train, args.n_val, args.n_test, args.split_seed,
            )
    else:
        weights_dir = Path(args.weights_dir)
        with (weights_dir / "train_result.json").open() as f:
            train_result = json.load(f)
        w_star, extra = load_weights(weights_dir / "trained_weights.npz")
        train_indices = extra["train_indices"].tolist()
        val_indices = extra.get("val_indices", np.array([])).tolist()
        test_indices = extra["test_indices"].tolist()
        T = len(w_star)

    k = args.budget_k if args.budget_k is not None else train_result.get("budget_k", 8)
    dummy_w = w_star if w_star is not None else np.zeros(T)
    selected_steps = discretize(dummy_w, k, disc_method, seed=disc_seed)
    k = len(selected_steps)
    print(f"[evaluate] T={T}, K={k}, method={disc_method}, selected_steps={selected_steps.tolist()}")

    run_cached_local = load_generated_run(run_dir, step_view="x0")
    prompts = [p for p in run_cached_local.prompts if p.num_steps == T]
    print(f"[evaluate] Loaded {len(prompts)} prompts")

    kernel = EmbeddingKernel(
        name=f"embedding-{args.kernel}",
        embedding_model=args.embedding_model,
        rbf_bandwidth=args.rbf_bandwidth,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        normalize_embeddings=True,
    )
    embedder = kernel.embedder

    entailment_model = None
    if not args.skip_nli:
        from src.semantic.entailment import load_entailment_model
        from src.features.sampling import CachedEntailmentModel
        print("[evaluate] Loading NLI model...")
        base_model = load_entailment_model(args.nli_model, device=args.nli_device)
        entailment_model = CachedEntailmentModel(base_model)

    from src.io.json_utils import read_jsonl, read_json
    examples_path = run_dir / "examples.jsonl"
    config_path = run_dir / "config.json"
    records = read_jsonl(examples_path) if examples_path.exists() else []
    config = read_json(config_path) if config_path.exists() else {}
    record_lookup = _build_record_lookup(records, config)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif weights_dir:
        output_dir = weights_dir
    else:
        output_dir = run_dir / "step_selection" / f"baseline_{disc_method}" / f"K_{k}"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_indices = {"train": train_indices, "val": val_indices, "test": test_indices}
    split_results = {}

    for split_name in args.splits:
        indices = split_indices[split_name]
        split_results[split_name] = evaluate_split(
            prompts, indices, selected_steps, k,
            kernel=kernel, embedder=embedder,
            entailment_model=entailment_model,
            record_lookup=record_lookup,
            split_name=split_name,
            nli_batch_size=args.nli_batch_size,
            use_greedy=args.use_greedy,
            trajectory_index=args.trajectory_index,
            skip_nli=args.skip_nli,
            full_traj=args.full_traj,
            w_star=w_star,
            discretization_method=disc_method,
            discretize_seed=disc_seed,
        )

    result = {
        "run_dir": str(run_dir),
        "weights_dir": str(weights_dir) if weights_dir else None,
        "T": T, "K": k,
        "selected_steps": selected_steps.tolist(),
        "discretization_method": disc_method,
        "is_baseline": is_baseline,
    }
    result.update(split_results)

    with (output_dir / "evaluation_result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2, allow_nan=True)

    print(f"\n[evaluate] Results saved to {output_dir / 'evaluation_result.json'}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate step selection quality")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--discretization_method", choices=["top_k", "randomized", "uniform", "random", "last_k"], default="top_k")
    parser.add_argument("--budget_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "cosine"])
    parser.add_argument("--rbf_bandwidth", default="median")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli_device", default=None)
    parser.add_argument("--nli_batch_size", type=int, default=8)
    parser.add_argument("--skip_nli", action="store_true")
    parser.add_argument("--use_greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--full_traj", action="store_true",
                        help="Also compute full-trajectory baseline features (full-*) alongside selected-*")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "val", "test"])
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.discretization_method not in BASELINE_METHODS and not args.weights_dir:
        raise SystemExit("--weights_dir is required for learned methods (top_k, randomized)")
    run_evaluate(args)


if __name__ == "__main__":
    main()
