"""CLI entry point: unified pipeline orchestrator.

Runs all stages in sequence: prepare -> generate -> label -> split -> features -> evaluate -> export.
Can also run individual stages via --stages.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.config import (
    MODEL_BACKENDS,
    MODEL_HF_IDS,
    Dataset,
    Model,
    Remasking,
    build_generation_config,
    config_filename,
    run_id as build_run_id,
    save_config,
)
from src.seed import seed_everything

ALL_STAGES = ("prepare", "generate", "label", "split", "features", "evaluate", "export")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the uncertainty-DLM pipeline.")
    parser.add_argument("--model", required=True, choices=[m.value for m in Model])
    parser.add_argument("--dataset", required=True, choices=[d.value for d in Dataset])
    parser.add_argument("--num_response_samples", type=int, default=20)
    parser.add_argument("--length", type=int, required=True, help="Generation length in tokens")
    parser.add_argument("--steps", type=int, required=True, help="Denoising steps")
    parser.add_argument("--remasking", required=True, choices=[r.value for r in Remasking])

    parser.add_argument("--fewshot_k", type=int, default=0)
    parser.add_argument("--stages", nargs="+", default=list(ALL_STAGES), help="Stages to run")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--generate_greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_questions", type=int, default=1000)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--judge_model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--judge_tp", type=int, default=None, help="vLLM tensor parallel size (default: auto)")
    parser.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli_batch_size", type=int, default=32)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)

    parser.add_argument("--train_split", type=int, default=10)
    parser.add_argument("--val_split", type=int, default=30)
    parser.add_argument("--test_split", type=int, default=40)

    parser.add_argument("--skip_mmd", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    model = Model(args.model)
    dataset = Dataset(args.dataset)
    remasking = Remasking(args.remasking)
    fewshot_k = args.fewshot_k
    rid = build_run_id(model, dataset, args.length, args.steps, remasking,
                        temperature=args.temperature if args.temperature != 1.0 else None,
                        fewshot_k=fewshot_k if fewshot_k > 0 else None)
    run_dir = Path("outputs") / rid
    config_path = Path("configs") / config_filename(model, dataset, args.length, args.steps, remasking,
                                                     fewshot_k=fewshot_k if fewshot_k > 0 else None)

    print(f"{'='*60}")
    print(f"Pipeline: {rid}")
    print(f"Model: {model.value} ({MODEL_BACKENDS[model]})")
    print(f"Dataset: {dataset.value}")
    print(f"Run dir: {run_dir}")
    print(f"Stages: {' -> '.join(args.stages)}")
    print(f"{'='*60}")

    # Generate config if needed
    if not config_path.exists():
        overrides = {}
        if args.batch_size:
            overrides["batch_size"] = args.batch_size
        config = build_generation_config(
            model, dataset, args.length, args.steps, remasking,
            num_questions=args.num_questions, temperature=args.temperature,
            num_response_samples=args.num_response_samples,
            generate_greedy=args.generate_greedy,
            fewshot_k=fewshot_k, **overrides,
        )
        config["run_id"] = rid
        save_config(config, config_path)
        print(f"Generated config: {config_path}")

    prompts_path = run_dir / "prompts.jsonl"
    features_path = run_dir / "uq_features.jsonl"
    metrics_json = run_dir / "uq_eval_metrics.json"
    metrics_csv = run_dir / "uq_eval_metrics.csv"

    if "prepare" in args.stages:
        print(f"\n{'='*60}\nStage: prepare\n{'='*60}")
        from src.cli.prepare import main as prepare_main
        prepare_main(["--config", str(config_path), "--output-root", "outputs"])

    if "generate" in args.stages:
        print(f"\n{'='*60}\nStage: generate\n{'='*60}")
        from src.cli.generate import main as generate_main
        gen_args = [
            "--config", str(config_path),
            "--run_id", rid,
            "--num_response_samples", str(args.num_response_samples),
            "--fewshot_k", str(fewshot_k),
        ]
        if prompts_path.exists():
            gen_args += ["--prompts", str(prompts_path)]
        gen_args.append("--generate_greedy" if args.generate_greedy else "--no-generate_greedy")
        if args.temperature is not None:
            gen_args += ["--temperature", str(args.temperature)]
        if args.num_questions:
            gen_args += ["--num_questions", str(args.num_questions)]
        if args.batch_size:
            gen_args += ["--batch_size", str(args.batch_size)]
        generate_main(gen_args)

    if "label" in args.stages:
        print(f"\n{'='*60}\nStage: label\n{'='*60}")
        from src.cli.label import main as label_main
        label_args = ["--run-dir", str(run_dir), "--print-examples"]
        label_args += ["--judge-model", args.judge_model]
        if args.judge_tp is not None:
            label_args += ["--judge-tp", str(args.judge_tp)]
        label_main(label_args)

    if "split" in args.stages:
        print(f"\n{'='*60}\nStage: split\n{'='*60}")
        from src.cli.split import main as split_main
        split_main([
            "--run-dir", str(run_dir),
            "--train", str(args.train_split),
            "--val", str(args.val_split),
            "--test", str(args.test_split),
            "--seed", str(args.seed),
        ])

    if "features" in args.stages:
        print(f"\n{'='*60}\nStage: features\n{'='*60}")
        from src.cli.features import main as features_main
        features_main([
            "--input_logs", str(run_dir),
            "--output", str(features_path),
            "--nli_model", args.nli_model,
            "--nli_batch_size", str(args.nli_batch_size),
            "--seed", str(args.seed),
        ])

    if "evaluate" in args.stages:
        print(f"\n{'='*60}\nStage: evaluate\n{'='*60}")
        from src.cli.evaluate import main as evaluate_main
        evaluate_main([
            "--features_path", str(features_path),
            "--output_metrics_json", str(metrics_json),
            "--output_metrics_csv", str(metrics_csv),
            "--bootstrap_samples", str(args.bootstrap_samples),
            "--seed", str(args.seed),
        ])

    if "export" in args.stages:
        print(f"\n{'='*60}\nStage: export\n{'='*60}")
        from src.cli.export import main as export_main
        xlsx_path = run_dir / "uq_results.xlsx"
        export_main([
            "--metrics-json", str(metrics_json),
            "--output", str(xlsx_path),
            "--empty-ok",
        ])

    print(f"\n{'='*60}")
    print(f"Pipeline finished: {rid}")
    print(f"Run dir: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
