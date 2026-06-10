"""Build prompts for every config JSON and save them to each run's output directory.

This script runs on CPU (no model/GPU needed). It loads each dataset, builds
raw prompts, and writes prompts.jsonl into the output directory that the
generation job will later use.

Usage:
    python scripts/prepare_all_prompts.py configs/LLaDA1.5_*.json
    python scripts/prepare_all_prompts.py --output-dir outputs/my_run configs/*.json
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import derive_run_id, load_config
from src.datasets.dataloader import build_raw_prompts, write_prompts_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompts.jsonl for each config.")
    parser.add_argument("configs", nargs="+", help="Config JSON file paths")
    parser.add_argument("--output-dir", default=None,
                        help="Parent output directory (default: outputs/<run_id>)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")

    for config_path in args.configs:
        config = load_config(config_path)
        run_id = config.get("run_id") or derive_run_id(config)

        if args.output_dir:
            out_dir = os.path.join(args.output_dir, run_id)
        else:
            out_dir = os.path.join("outputs", run_id)

        prompts_path = os.path.join(out_dir, "prompts.jsonl")
        if os.path.isfile(prompts_path):
            print(f"  SKIP (exists): {prompts_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Config: {config_path}")
        print(f"Run ID: {run_id}")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")

        try:
            dataset_key, qa_samples, raw_prompts = build_raw_prompts(config, hf_token=hf_token)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        write_prompts_jsonl(prompts_path, dataset_key, qa_samples, raw_prompts)
        print(f"  Wrote {len(qa_samples)} prompts to {prompts_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
