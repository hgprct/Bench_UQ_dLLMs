"""CLI: build raw prompts for each dataset and save them to JSONL files."""

from __future__ import annotations

import argparse
import os

from src.config import Dataset
from src.datasets.dataloader import build_raw_prompts, write_prompts_jsonl


ALL_DATASETS = [ds.value for ds in Dataset]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prompts for each dataset and save to JSONL.",
    )
    parser.add_argument(
        "-n", "--num-prompts", type=int, default=5,
        help="Number of prompts to build per dataset (default: 5)",
    )
    parser.add_argument(
        "-d", "--datasets", nargs="*", default=None,
        choices=ALL_DATASETS,
        help=f"Datasets to process (default: all). Choices: {ALL_DATASETS}",
    )
    parser.add_argument(
        "-o", "--output-dir", default="outputs/prompts_check",
        help="Output directory (default: outputs/prompts_check)",
    )
    parser.add_argument(
        "--fewshot-k", type=int, default=0,
        help="Number of few-shot examples (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = args.datasets or ALL_DATASETS
    os.makedirs(args.output_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", "")

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  (n={args.num_prompts})")
        print(f"{'='*60}")

        config = {
            "dataset": ds_name,
            "num_questions": args.num_prompts,
            "fewshot_k": args.fewshot_k,
        }

        try:
            dataset_key, qa_samples, raw_prompts = build_raw_prompts(config, hf_token=hf_token)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        out_path = os.path.join(args.output_dir, f"{ds_name}_prompts.jsonl")
        write_prompts_jsonl(out_path, dataset_key, qa_samples, raw_prompts)

        txt_path = os.path.join(args.output_dir, f"{ds_name}_prompts.txt")
        with open(txt_path, "w") as f:
            f.write(f"Dataset: {ds_name}  |  {len(qa_samples)} prompts  |  fewshot_k={args.fewshot_k}\n")
            f.write("=" * 70 + "\n\n")
            for i, prompt in enumerate(raw_prompts):
                f.write(f"--- prompt {i} ---\n")
                f.write(f"{prompt}\n\n")
        print(f"  Wrote {txt_path}")

        for i, (sample, prompt) in enumerate(zip(qa_samples, raw_prompts)):
            print(f"\n  --- prompt {i} ---")
            print(f"  question: {str(sample.get('question', ''))[:120]}")
            print(f"  prompt:   {prompt[:200]}")

    print(f"\nDone. Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
