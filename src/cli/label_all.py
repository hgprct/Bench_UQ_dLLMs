"""CLI: label all runs in a folder, loading the LLM judge once for all GPU-mode runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label every run in a folder. Loads the LLM judge once for all GPU-mode runs."
    )
    parser.add_argument("--folder", default="outputs/", help="Folder containing run directories")
    parser.add_argument("--judge-model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--judge-tp", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--judge-max-num-seqs", type=int, default=None, help="vLLM max concurrent sequences")
    parser.add_argument("--judge-gpu-mem", type=float, default=None, help="vLLM GPU memory utilization (0-1)")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="RUN", help="Run directory names to skip")
    parser.add_argument("--print-examples", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-label all records even if already labeled")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _discover_runs(folder: Path, exclude: set[str]) -> list[Path]:
    runs = []
    for d in sorted(folder.iterdir()):
        if not d.is_dir():
            continue
        if d.name in exclude:
            print(f"[label_all] Skipping excluded: {d.name}")
            continue
        if (d / "config.json").exists() and (d / "answers.jsonl").exists():
            runs.append(d)
    return runs


def _print_accuracy(records: list[dict], name: str) -> None:
    correct = sum(1 for r in records if r.get("label") is True)
    total = sum(1 for r in records if r.get("label") is not None)
    if total > 0:
        print(f"[label_all] {name} accuracy: {correct}/{total} = {correct / total:.3f}")
    else:
        print(f"[label_all] {name}: no labels found")


def _label_run(
    run_dir: Path,
    evaluator,
    judge_model: str,
    judge_tp: int | None,
    print_examples: bool,
    dataset_key: str,
    force: bool = False,
) -> None:
    from src.utils.io import read_jsonl
    from src.config import DATASET_CONFIGS, Dataset
    from src.labeling import label_records

    ds_config = DATASET_CONFIGS[Dataset(dataset_key)]
    records = read_jsonl(run_dir / "answers.jsonl")

    print(f"[label_all] {run_dir.name}: {len(records)} records, method='{ds_config.label_method}'")

    labeled = label_records(
        records, dataset_key,
        judge_model=judge_model,
        judge_tp=judge_tp,
        evaluator=evaluator,
        force=force,
    )

    with open(run_dir / "answers.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

    print(f"[label_all] Labeled {labeled} records. Updated {run_dir / 'answers.jsonl'}")
    if print_examples:
        _print_accuracy(records, run_dir.name)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    folder = Path(args.folder)
    exclude = set(args.exclude or [])

    runs = _discover_runs(folder, exclude)
    if not runs:
        print(f"[label_all] No valid run directories found in {folder}")
        return

    print(f"[label_all] Found {len(runs)} run(s) to label")

    gpu_runs: list[tuple[Path, str]] = []
    cpu_runs: list[tuple[Path, str]] = []

    from src.config import DATASET_CONFIGS, Dataset
    for run_dir in runs:
        with open(run_dir / "config.json") as f:
            config = json.load(f)
        dataset_key = config.get("dataset", "triviaqa")
        ds_config = DATASET_CONFIGS[Dataset(dataset_key)]
        if ds_config.label_gpu_mode == "gpu":
            gpu_runs.append((run_dir, dataset_key))
        else:
            cpu_runs.append((run_dir, dataset_key))

    print(f"[label_all] cpu runs: {len(cpu_runs)}, gpu (judge) runs: {len(gpu_runs)}")

    for run_dir, dataset_key in cpu_runs:
        _label_run(run_dir, None, args.judge_model, args.judge_tp, args.print_examples, dataset_key, force=args.force)

    evaluator = None
    for run_dir, dataset_key in gpu_runs:
        if evaluator is None:
            from src.judge.evaluator import EvaluatorLLMLocal
            print(f"[label_all] Loading judge: {args.judge_model}")
            evaluator = EvaluatorLLMLocal(
                model_name=args.judge_model,
                tensor_parallel_size=args.judge_tp,
                max_num_seqs=args.judge_max_num_seqs,
                gpu_memory_utilization=args.judge_gpu_mem,
            )
        _label_run(run_dir, evaluator, args.judge_model, args.judge_tp, args.print_examples, dataset_key, force=args.force)

    print("[label_all] All runs labeled.")


if __name__ == "__main__":
    main()
