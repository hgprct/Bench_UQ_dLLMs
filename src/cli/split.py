"""CLI entry point for Stage 3: Train/val/test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.io.json_utils import read_jsonl
from src.split import save_splits, split_prompts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split prompts into train/val/test sets.")
    parser.add_argument("--run-dir", required=True, help="Run directory with examples.jsonl")
    parser.add_argument("--train", type=int, default=10, help="Number of training prompts")
    parser.add_argument("--val", type=int, default=30, help="Number of validation prompts")
    parser.add_argument("--test", type=int, default=40, help="Number of test prompts")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    records = read_jsonl(run_dir / "examples.jsonl")

    prompt_ids = sorted({
        str(r.get("qa_example_id", r.get("example_id", i)))
        for i, r in enumerate(records)
    })
    print(f"[split] Found {len(prompt_ids)} unique prompts")

    from src.datasets.labeling import unlabeled_prompt_ids
    excluded = unlabeled_prompt_ids(records)
    if excluded:
        prompt_ids = [pid for pid in prompt_ids if pid not in excluded]
        print(f"[split] Excluded {len(excluded)} prompt(s) with ambiguous labels, {len(prompt_ids)} remain")

    splits = split_prompts(
        prompt_ids, train=args.train, val=args.val, test=args.test, seed=args.seed,
    )
    output = run_dir / "splits.json"
    save_splits(splits, output, seed=args.seed, train=args.train, val=args.val, test=args.test)
    print(f"[split] Wrote {output}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
