"""CLI entry point: prepare prompts from dataset configs (no GPU required)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import derive_run_id, load_config
from src.generate.dataset_inputs import build_prompt_records, write_prompts_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare prompts.jsonl from dataset configs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", nargs="+", help="Path(s) to config JSON file(s)")
    group.add_argument("--all", action="store_true", help="Process all configs in --config-dir")
    parser.add_argument("--config-dir", default="configs", help="Directory containing config JSONs (for --all)")
    parser.add_argument("--output-root", default="outputs", help="Root output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prompts.jsonl")
    return parser.parse_args(argv)


def _resolve_config_paths(args: argparse.Namespace) -> list[Path]:
    """Determine which config paths to process based on CLI arguments."""
    if args.config:
        return [Path(p) for p in args.config]
    config_dir = Path(args.config_dir)
    paths = sorted(config_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No config files found in {config_dir}")
    return paths


def _prepare_one(config_path: Path, output_root: str, overwrite: bool) -> None:
    """Prepare prompts.jsonl for a single config."""
    config = load_config(config_path)
    run_id = str(config.get("run_id") or derive_run_id(config))
    output_dir = os.path.join(output_root, run_id) # Output directory for this run
    prompts_path = os.path.join(output_dir, "prompts.jsonl") # Path for prompts

    if os.path.isfile(prompts_path) and not overwrite:
        print(f"  SKIP (exists): {prompts_path}")
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"  Preparing: {config_path} -> {prompts_path}")
    records = build_prompt_records(config, hf_token or None) # Create prompts
    write_prompts_jsonl(records, prompts_path) # Save prompts to file


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_paths = _resolve_config_paths(args)

    print(f"Preparing prompts for {len(config_paths)} config(s)...")
    for config_path in config_paths: #Handle each config one by one
        _prepare_one(config_path, args.output_root, args.overwrite)

    print(f"\nDone. Processed {len(config_paths)} config(s).")


if __name__ == "__main__":
    main()
