#!/usr/bin/env python3
"""Trial script: generate one greedy answer for N prompts of each dataset.

Loads the model once, then iterates over all registered datasets (or a
user-supplied subset). Only final answers are collected; no trajectory
data is saved.

Usage:
    python scripts/trial_generate.py [--model_id ID] [--num_prompts N]
        [--gen_length L] [--steps S] [--datasets d1 d2 ...]
        [--output path/to/results.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best = i
    return torch.device(f"cuda:{best}")


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.config import Dataset, MODEL_HF_IDS, Model

    all_datasets = [d.value for d in Dataset]
    default_model = MODEL_HF_IDS[Model.LLaDA]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=default_model,
                        help=f"HuggingFace model ID (default: {default_model})")
    parser.add_argument("--num_prompts", type=int, default=10,
                        help="Number of prompts per dataset (default: 10)")
    parser.add_argument("--gen_length", type=int, default=128,
                        help="Generation length in tokens (default: 128)")
    parser.add_argument("--steps", type=int, default=128,
                        help="Denoising steps (default: 128)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for generation (default: 10)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        choices=all_datasets, metavar="DATASET",
                        help=f"Datasets to run (default: all). Choices: {all_datasets}")
    parser.add_argument("--output", default=None,
                        help="Optional path to write results as JSONL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from src.config import Dataset, MODEL_BACKENDS, MODEL_HF_IDS, Model, resolve_remasking
    from src.datasets.dataloader import prepare_dataset_inputs
    from src.generate import generate as generate_fn
    from src.generate.model import (
        infer_eos_token_ids, infer_mask_token_id,
        load_model, load_tokenizer, pad_token_id as get_pad_token_id,
    )
    from src.seed import seed_everything

    seed_everything(42)
    device = _select_device()
    print(f"Device: {device}")

    # Resolve backend from model_id
    hf_id_to_model = {v: k for k, v in MODEL_HF_IDS.items()}
    model_enum = hf_id_to_model.get(args.model_id)
    backend = MODEL_BACKENDS.get(model_enum, "llada") if model_enum else "llada"
    if "dream" in args.model_id.lower():
        backend = "dream"
    elif "nemotron" in args.model_id.lower():
        backend = "nemotron"

    print(f"Model:   {args.model_id}")
    print(f"Backend: {backend}")

    # Load model and tokenizer once
    model = load_model(args.model_id, device)
    tokenizer = load_tokenizer(args.model_id)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    eos_token_ids = infer_eos_token_ids(tokenizer, model_id=args.model_id)

    pad_id = get_pad_token_id(tokenizer)
    if pad_id is None:
        tokenizer.pad_token_id = eos_token_ids[0]

    mask_id = infer_mask_token_id(tokenizer, model=model)
    if mask_id is None:
        raise RuntimeError("Could not infer mask_token_id from model/tokenizer")

    datasets_to_run = args.datasets or [d.value for d in Dataset]
    hf_token = os.environ.get("HF_TOKEN", "")

    all_results: list[dict] = []
    failed: list[str] = []

    for ds_name in datasets_to_run:
        _print_section(f"Dataset: {ds_name}  (n={args.num_prompts})")

        generation_config = {
            "dataset": ds_name,
            "num_questions": args.num_prompts,
            "fewshot_k": 0,
        }

        try:
            dataset_key, qa_pairs, prompts = prepare_dataset_inputs(
                generation_config, tokenizer, hf_token,
            )
        except Exception as exc:
            print(f"  ERROR loading dataset: {exc}")
            failed.append(ds_name)
            continue

        print(f"  Loaded {len(prompts)} prompts")

        try:
            answers, _ = generate_fn(
                model=model,
                prompts=prompts,
                device=device,
                backend=backend,
                batch_size=args.batch_size,
                tokenizer=tokenizer,
                steps=args.steps,
                gen_length=args.gen_length,
                temperature=0.0,
                remasking=resolve_remasking("lc"),
                mask_id=mask_id,
                eos_token_ids=eos_token_ids,
                save_trajectory=False,
            )
        except Exception as exc:
            print(f"  ERROR during generation: {exc}")
            failed.append(ds_name)
            continue

        for i, (qa, answer) in enumerate(zip(qa_pairs, answers)):
            record = {
                "dataset": dataset_key,
                "prompt_idx": i,
                "question": qa.get("question", "") if isinstance(qa, dict) else "",
                "reference_answer": qa.get("reference_answer", "") if isinstance(qa, dict) else "",
                "answer": answer,
            }
            all_results.append(record)
            ref = str(record["reference_answer"])[:60]
            print(f"  [{i:02d}] Q: {str(record['question'])[:60]}")
            print(f"       A: {answer[:120]}")
            print(f"       ref: {ref}")

    # Summary
    _print_section("Summary")
    print(f"Total answers: {len(all_results)}")
    if failed:
        print(f"Failed datasets: {', '.join(failed)}")

    # Save if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            for rec in all_results:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
