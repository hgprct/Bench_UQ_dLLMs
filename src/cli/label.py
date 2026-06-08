"""CLI entry point for Stage 2: Label correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label greedy final answers as correct/incorrect.")
    parser.add_argument("--run-dir", required=True, nargs="+", help="Run directory(ies) with examples.jsonl + config.json")
    parser.add_argument("--method", default=None, help="Labeling method: exact_match or llm_judge (default: from config)")
    parser.add_argument("--judge-model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--judge-tp", type=int, default=None, help="vLLM tensor parallel size (default: auto)")
    parser.add_argument("--judge-max-num-seqs", type=int, default=None, help="vLLM max concurrent sequences")
    parser.add_argument("--judge-gpu-mem", type=float, default=None, help="vLLM GPU memory utilization (0-1)")
    parser.add_argument("--print-examples", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-label all records even if already labeled")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    run_dirs = [Path(d) for d in args.run_dir]
    evaluator = None

    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        examples_path = run_dir / "examples.jsonl"

        if not examples_path.exists() or not config_path.exists():
            print(f"[label] Skipping {run_dir.name}: missing config.json or examples.jsonl")
            continue

        with open(config_path) as f:
            config = json.load(f)

        from src.io.json_utils import read_jsonl
        records = read_jsonl(examples_path)

        dataset_key = config.get("dataset", "triviaqa")
        from src.registry import get_dataset_module
        dataset_module = get_dataset_module(dataset_key)

        from src.config import Dataset, DATASET_DEFAULTS
        ds_default_method = DATASET_DEFAULTS[Dataset(dataset_key)].default_label_method
        method = args.method or ds_default_method
        print(f"[label] {run_dir.name}: {len(records)} records, method='{method}'")

        if method == "llm_judge" and evaluator is None:
            from src.judge.evaluator import EvaluatorLLMLocal
            evaluator = EvaluatorLLMLocal(
                model_name=args.judge_model,
                tensor_parallel_size=args.judge_tp,
                max_num_seqs=args.judge_max_num_seqs,
                gpu_memory_utilization=args.judge_gpu_mem,
            )

        from src.datasets.labeling import label_records
        labeled = label_records(
            records, dataset_module, method,
            judge_model=args.judge_model,
            judge_tp=args.judge_tp,
            evaluator=evaluator,
            force=args.force,
        )

        with open(examples_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

        print(f"[label] Labeled {labeled} records. Updated {examples_path}")

        from src.datasets.labeling import unlabeled_prompt_ids
        excluded = unlabeled_prompt_ids(records)
        if excluded:
            sorted_ids = sorted(excluded)
            preview = sorted_ids[:10]
            print(f"[label] WARNING: {len(excluded)} prompt(s) could not be labeled "
                  f"(judge output unparseable after retry).")
            print(f"[label]   Affected prompt IDs: {preview}"
                  + (f" ... ({len(excluded)} total)" if len(excluded) > 10 else ""))
            print(f"[label]   These prompts will be excluded from downstream evaluation.")

        if args.print_examples:
            correct = sum(1 for r in records if r.get("final", {}).get("is_correct") is True)
            incorrect = sum(1 for r in records if r.get("final", {}).get("is_correct") is False)
            ambiguous = sum(1 for r in records if r.get("final", {}).get("is_correct") is None
                           and "correctness_method" in r.get("final", {}))
            total = correct + incorrect
            print(f"[label] Results: {correct} correct, {incorrect} incorrect"
                  + (f", {ambiguous} ambiguous" if ambiguous else "")
                  + (f" — accuracy {correct/total:.3f}" if total > 0 else ""))


if __name__ == "__main__":
    main()
