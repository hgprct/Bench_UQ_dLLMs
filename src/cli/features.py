"""CLI entry point for Stage 4: Compute UQ features."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build light UQ features from a generation run.")
    parser.add_argument("--input_logs", required=True, help="Run directory with examples.jsonl + traces.npz")
    parser.add_argument("--output", required=True, help="Output path for uq_features.jsonl")
    parser.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli_batch_size", type=int, default=32)
    parser.add_argument("--sample_nli_max_items", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    from src.semantic.entailment import load_entailment_model
    from src.labeling import unlabeled_prompt_ids
    from src.features.builder import build_feature_rows
    from src.utils.io import read_jsonl, write_jsonl

    run_dir = Path(args.input_logs)
    records = read_jsonl(run_dir / "examples.jsonl")
    excluded = unlabeled_prompt_ids(records)
    if excluded:
        print(f"[features] Excluding {len(excluded)} prompt(s) with ambiguous labels")

    print(f"Loading NLI model: {args.nli_model}")
    nli_model = load_entailment_model(args.nli_model)

    sample_max = args.sample_nli_max_items if args.sample_nli_max_items > 0 else None
    rows = build_feature_rows(
        args.input_logs,
        nli_model=nli_model,
        nli_batch_size=args.nli_batch_size,
        sample_nli_max_items=sample_max,
        progress_every=args.progress_every,
    )

    if excluded:
        rows = [r for r in rows if str(r["prompt_id"]) not in excluded]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} feature rows to {output_path}")


if __name__ == "__main__":
    main()
