"""CLI entry point for generating config files from the reference template."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    Dataset,
    Model,
    Remasking,
    build_generation_config,
    config_filename,
    save_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate config files for all model/dataset/remasking combos.")
    parser.add_argument("--model", nargs="+", default=[m.value for m in Model],
                        choices=[m.value for m in Model])
    parser.add_argument("--dataset", nargs="+", default=[d.value for d in Dataset],
                        choices=[d.value for d in Dataset])
    parser.add_argument("--remasking", nargs="+", default=[r.value for r in Remasking],
                        choices=[r.value for r in Remasking])
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num_questions", type=int, default=1000)
    parser.add_argument("--num_response_samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--generate_greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fewshot_k", type=int, default=0)
    parser.add_argument("--confidence_eos_eot_inf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="configs")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for model_name in args.model:
        model = Model(model_name)
        for dataset_name in args.dataset:
            dataset = Dataset(dataset_name)
            for rm_name in args.remasking:
                remasking = Remasking(rm_name)
                fname = config_filename(model, dataset, args.length, args.steps, remasking,
                                       fewshot_k=args.fewshot_k,
                                       confidence_eos_eot_inf=args.confidence_eos_eot_inf)
                path = output_dir / fname

                if path.exists() and not args.overwrite:
                    print(f"  SKIP (exists): {path}")
                    continue

                config = build_generation_config(
                    model, dataset, args.length, args.steps, remasking,
                    num_questions=args.num_questions,
                    temperature=args.temperature,
                    num_response_samples=args.num_response_samples,
                    generate_greedy=args.generate_greedy,
                    fewshot_k=args.fewshot_k,
                    confidence_eos_eot_inf=args.confidence_eos_eot_inf,
                )
                save_config(config, path)
                print(f"  WROTE: {path}")
                count += 1

    print(f"\nGenerated {count} config file(s) in {output_dir}")


if __name__ == "__main__":
    main()
