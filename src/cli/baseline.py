"""CLI entry point: compute all baseline UQ features for every prompt.

Produces baseline_features.jsonl with token features (msp, perplexity, mte),
iid-sample features (mcnse, se-marginal, ...), full-trajectory AD (full-ad),
full-trajectory NLI features (full-se-marginal, ..., optional via --full_scope),
and random-scope features (random-{i}-se-marginal, ...) computed on K=20 randomly
picked steps for each of 5 random selections (metrics averaged downstream).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.seed import seed_everything

RANDOM_SCOPE_K = 20
NUM_RANDOM_SELECTIONS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute baseline UQ features (token + iid-sample + full-trajectory) for all prompts.",
    )
    parser.add_argument("--run_dir", required=True, help="Run directory with answers.jsonl, topk_logprobs.npz, config.json")
    parser.add_argument("--output", default=None, help="Output path (default: <run_dir>/baseline_features.jsonl)")
    parser.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli_batch_size", type=int, default=512)
    parser.add_argument("--sample_nli_max_items", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_scope", action="store_true",
                        help="Also compute full-trajectory NLI features (slow; skipped by default)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    from src.features.builder import build_feature_rows
    from src.features.sampling import CachedEntailmentModel
    from src.labeling import unlabeled_prompt_ids
    from src.utils.io import read_jsonl, write_jsonl
    from src.semantic.entailment import load_entailment_model

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "baseline_features.jsonl"

    records = read_jsonl(run_dir / "answers.jsonl")
    excluded = unlabeled_prompt_ids(records)
    if excluded:
        print(f"[baseline] Excluding {len(excluded)} prompt(s) with ambiguous labels")

    print(f"[baseline] Loading NLI model: {args.nli_model}")
    nli_model = load_entailment_model(args.nli_model)

    print("[baseline] Computing token + iid-sample features for all prompts...")
    sample_max = args.sample_nli_max_items if args.sample_nli_max_items > 0 else None
    print(f"[baseline] nli_batch_size={args.nli_batch_size}, sample_nli_max_items={sample_max}")
    rows = build_feature_rows(
        run_dir,
        nli_model=nli_model,
        nli_batch_size=args.nli_batch_size,
        sample_nli_max_items=sample_max,
        progress_every=args.progress_every,
    )
    if excluded:
        rows = [r for r in rows if str(r["prompt_id"]) not in excluded]
    rows_by_pid = {str(r["prompt_id"]): r for r in rows}

    from collections import Counter
    import numpy as np
    from src.features.directions import NLI_FEATURES
    from src.features.trajectory import compute_averaged_dissimilarity
    from src.mmd.load import load_generated_run
    from src.selection.evaluate import compute_trajectory_features

    print("[baseline] Loading trajectories...")
    run = load_generated_run(run_dir, step_view="x0")
    prompts = list(run.prompts)
    T = Counter(p.num_steps for p in prompts).most_common(1)[0][0]
    prompts = [p for p in prompts if p.num_steps == T]
    if excluded:
        prompts = [p for p in prompts if p.prompt_id not in excluded]
    print(f"[baseline] {len(prompts)} prompts with T={T}")

    cached_nli = CachedEntailmentModel(nli_model)

    if args.full_scope:
        _prefetch_full_trajectory_nli(
            prompts, cached_nli,
            rows_by_pid=rows_by_pid,
            nli_batch_size=args.nli_batch_size,
        )
    else:
        _prefetch_ad_nli(
            prompts, cached_nli,
            rows_by_pid=rows_by_pid,
            nli_batch_size=args.nli_batch_size,
        )

    n_ad = 0
    n_full = 0
    n_random = 0

    for count, prompt in enumerate(prompts):
        row = rows_by_pid.get(prompt.prompt_id)
        if row is None:
            continue

        samples = sorted(prompt.samples, key=lambda s: s.response_sample_id)
        target = samples[0] if samples else None

        if target is not None:
            ad_val = compute_averaged_dissimilarity(
                target.step_answers, target.final_answer, cached_nli,
                nli_batch_size=args.nli_batch_size,
            )
            row["full-ad"] = ad_val
            if ad_val is not None:
                n_ad += 1
        else:
            row["full-ad"] = None

        if args.full_scope:
            if target is not None:
                all_steps = np.arange(len(target.step_answers))
                full_feats = compute_trajectory_features(
                    target, all_steps, cached_nli,
                    prefix="full", nli_batch_size=args.nli_batch_size,
                )
                row.update(full_feats)
                n_full += 1
            else:
                for f in NLI_FEATURES:
                    row[f"full-{f}"] = None

        if target is not None:
            random_feats = _compute_random_scope_features(
                target, cached_nli,
                k=RANDOM_SCOPE_K,
                num_selections=NUM_RANDOM_SELECTIONS,
                nli_batch_size=args.nli_batch_size,
                seed=args.seed,
                prompt_index=count,
            )
            row.update(random_feats)
            n_random += 1
        else:
            for rep in range(NUM_RANDOM_SELECTIONS):
                for f in NLI_FEATURES:
                    row[f"random-{rep}-{f}"] = None

        if args.progress_every and (count + 1) % args.progress_every == 0:
            parts = ["full-ad", "random-scope"]
            if args.full_scope:
                parts.insert(1, "full-trajectory")
            label = " + ".join(parts)
            print(f"[baseline] {label}: {count + 1}/{len(prompts)} prompts")

    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    parts = [f"{n_ad} full-ad", f"{n_random} random-scope"]
    if n_full:
        parts.insert(1, f"{n_full} full-trajectory")
    extra = f" ({', '.join(parts)})"
    print(f"[baseline] Wrote {len(rows)} rows{extra} to {output}")


def _compute_random_scope_features(
    target,
    cached_nli,
    *,
    k: int,
    num_selections: int,
    nli_batch_size: int,
    seed: int,
    prompt_index: int,
) -> dict[str, float | None]:
    """Compute multi-sample UQ features for each random step selection separately."""
    import numpy as np
    from src.features.directions import NLI_FEATURES
    from src.selection.evaluate import compute_trajectory_features

    n_steps = len(target.step_answers)
    actual_k = min(k, n_steps)

    result: dict[str, float | None] = {}
    for rep in range(num_selections):
        rng = np.random.default_rng(seed * 100_000 + prompt_index * 1_000 + rep)
        random_steps = np.sort(rng.choice(n_steps, size=actual_k, replace=False))
        feats = compute_trajectory_features(
            target, random_steps, cached_nli,
            prefix="random", nli_batch_size=nli_batch_size,
        )
        for fname, val in feats.items():
            indexed = fname.replace("random-", f"random-{rep}-", 1)
            result[indexed] = val

    for rep in range(num_selections):
        for f in NLI_FEATURES:
            result.setdefault(f"random-{rep}-{f}", None)
    return result


def _prefetch_ad_nli(
    prompts,
    cached_nli,
    *,
    rows_by_pid: dict,
    nli_batch_size: int,
) -> None:
    """Prefetch NLI pairs for averaged dissimilarity (step vs final answer)."""
    from src.mmd.load import is_valid_text

    all_premises: list[str] = []
    all_hypotheses: list[str] = []

    for prompt in prompts:
        if prompt.prompt_id not in rows_by_pid:
            continue
        samples = sorted(prompt.samples, key=lambda s: s.response_sample_id)
        target = samples[0] if samples else None
        if target is None or not is_valid_text(target.final_answer):
            continue

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
        print(f"[baseline] Prefetching {len(all_premises)} NLI pairs for AD...")
        cached_nli.batch_probabilities(all_premises, all_hypotheses, batch_size=nli_batch_size)


def _prefetch_full_trajectory_nli(
    prompts,
    cached_nli,
    *,
    rows_by_pid: dict,
    nli_batch_size: int,
) -> None:
    """Collect all step-answer NLI pairs across prompts and warm the cache in one GPU pass."""
    from src.mmd.load import is_valid_text

    all_premises: list[str] = []
    all_hypotheses: list[str] = []

    for prompt in prompts:
        if prompt.prompt_id not in rows_by_pid:
            continue
        samples = sorted(prompt.samples, key=lambda s: s.response_sample_id)
        target = samples[0] if samples else None
        if target is None:
            continue

        texts = list(dict.fromkeys(
            target.step_answers[i].strip()
            for i in range(len(target.step_answers))
            if is_valid_text(target.step_answers[i])
        ))
        if is_valid_text(target.final_answer):
            final_text = target.final_answer.strip()
            if final_text not in texts:
                texts.append(final_text)
        if len(texts) < 2:
            continue

        for left in texts:
            for right in texts:
                all_premises.append(left)
                all_hypotheses.append(right)

    if all_premises:
        print(f"[baseline] Prefetching {len(all_premises)} NLI pairs for full-trajectory + AD features...")
        cached_nli.batch_probabilities(all_premises, all_hypotheses, batch_size=nli_batch_size)


if __name__ == "__main__":
    main()
