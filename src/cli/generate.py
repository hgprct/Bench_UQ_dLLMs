"""CLI entry point for Stage 1: Generate QA traces."""

from __future__ import annotations

import argparse
import json
import os

import torch

from src.generate import generate as generate_fn
from src.generate.traces import save_trace_collection
from src.config import derive_run_id, load_config, resolve_remasking
from src.seed import seed_everything
from src.generate.dataset_inputs import load_prompts_jsonl, prepare_dataset_inputs

from src.generate.dataset_inputs import (
    expand_greedy_and_sampled, expand_response_samples, interleave_traces,
)

from src.generate.model import (
    infer_eos_token_ids, infer_mask_token_id,
    load_model, load_tokenizer, pad_token_id as get_pad_token_id,
)


_CLI_OVERRIDE_KEYS = (
    "num_questions", "num_response_samples", "batch_size", "steps",
    "gen_length", "temperature", "generate_greedy",
    "remasking", "mask_id",
    "topk_trace_k", "fewshot_k", "confidence_eos_eot_inf", "seed",
)

_CONFIG_DEFAULTS = dict(
    model_id="GSAI-ML/LLaDA-8B-Instruct",
    seed=42, steps=128, gen_length=128, temperature=0.0,
    remasking="lc", batch_size=8, topk_trace_k=64,
    num_response_samples=20, generate_greedy=True,
    num_questions=1000, fewshot_k=0,
    logits_eos_inf=False, confidence_eos_eot_inf=False,
)

def select_device() -> torch.device:
    """Select the CUDA device with the most free memory, or CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if torch.cuda.device_count() == 1:
        return torch.device("cuda:0")
    best_device = 0
    best_free = 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_device = i
    return torch.device(f"cuda:{best_device}")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dLLM QA traces.")
    parser.add_argument("--config", required=True, help="Path to generation config JSON")
    parser.add_argument("--run_id", help="Run ID for output directory naming")
    parser.add_argument("--num_questions", type=int)
    parser.add_argument("--num_response_samples", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--gen_length", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--generate_greedy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--remasking", choices=["lc", "rd"])
    parser.add_argument("--mask_id", type=int)
    parser.add_argument("--topk_trace_k", type=int)
    parser.add_argument("--fewshot_k", type=int)
    parser.add_argument("--confidence_eos_eot_inf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompts", help="Path to prompts.jsonl (from prepare stage)")
    parser.add_argument("--output_dir")
    return parser.parse_args(argv)


def _apply_config(config: dict, args: argparse.Namespace) -> None:
    """Apply CLI overrides and set defaults in the config dictionary."""
    for key in _CLI_OVERRIDE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
    if args.run_id:
        config["run_id"] = args.run_id
    for key, default in _CONFIG_DEFAULTS.items():
        config.setdefault(key, default)
    model_id = config["model_id"]
    backend = config.get("model_backend", "llada")
    if "dream" in model_id.lower():
        backend = "dream"
    elif "nemotron" in model_id.lower():
        backend = "nemotron"
    config["model_backend"] = backend


def _setup_model(config: dict, device):
    """Load the model and tokenizer, and infer EOS and mask token IDs."""

    model_id = config["model_id"]
    model = load_model(model_id, device) # Load the model
    tokenizer = load_tokenizer(model_id)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    eos_token_ids = infer_eos_token_ids(tokenizer, model_id=model_id)
    ptid = get_pad_token_id(tokenizer)
    if ptid is None:
        if eos_token_ids:
            tokenizer.pad_token_id = eos_token_ids[0]
            ptid = tokenizer.pad_token_id
            print(f"[DLM] No pad_token_id; using EOS id {ptid}")
        else:
            raise ValueError("No pad_token_id and no EOS tokens available")

    if config.get("mask_id") is None:
        config["mask_id"] = infer_mask_token_id(tokenizer, model=model)
    if config.get("mask_id") is None:
        raise ValueError("mask_id could not be inferred; set it in the config")
    config["eos_token_ids"] = eos_token_ids

    return model, tokenizer, eos_token_ids, ptid


def _load_or_prepare_inputs(config, args, tokenizer, output_dir):
    """Load prompts from file if available, otherwise prepare them from the dataset config."""
    prompts_path = args.prompts or os.path.join(output_dir, "prompts.jsonl")
    if os.path.isfile(prompts_path):
        print(f"Loading prepared prompts from {prompts_path}...")
        dataset_key, qa_pairs, prompts = load_prompts_jsonl(prompts_path, tokenizer)
        print(f"Loaded {len(qa_pairs)} prompts from '{dataset_key}'.")
    else:
        fewshot_k = int(config["fewshot_k"])
        if fewshot_k > 0:
            print(f"[DLM] Few-shot: {fewshot_k} examples")
        hf_token = os.environ.get("HF_TOKEN", "")
        print("Preparing dataset inputs inline...")
        dataset_key, qa_pairs, prompts = prepare_dataset_inputs(config, tokenizer, hf_token)
        print(f"Prepared {len(qa_pairs)} prompts from '{dataset_key}'.")
    return dataset_key, qa_pairs, prompts


def _run_generation(generate_fn, qa_pairs, prompts, config, common_kwargs, seed):
    """Run the generation function with the appropriate parameters based on the config."""

    num_sampled = int(config["num_response_samples"])
    temperature = float(config["temperature"])
    generate_greedy = bool(config["generate_greedy"])

    if temperature > 0:
        if num_sampled <= 0:
            raise ValueError("num_response_samples must be > 0 when temperature > 0")

        if generate_greedy:
            total = 1 + num_sampled
            print(f"[DLM] Two-pass: 1 greedy + {num_sampled} sampled (T={temperature})")

            greedy_answers, greedy_traces = generate_fn(prompts=prompts, temperature=0.0, **common_kwargs)

            seed_everything(seed + 1)
            sampled_qa, sampled_prompts = expand_response_samples(qa_pairs, prompts, num_sampled)
            sampled_answers, sampled_traces = generate_fn(prompts=sampled_prompts, temperature=temperature, **common_kwargs)

            rich_traces = interleave_traces(greedy_traces, sampled_traces, len(qa_pairs), num_sampled)
            merged_qa, merged_prompts = expand_greedy_and_sampled(qa_pairs, prompts, num_sampled)
            all_answers = _interleave_answers(greedy_answers, sampled_answers, len(qa_pairs), num_sampled)
            config["num_response_samples"] = total
        else:
            print(f"[DLM] Sampled-only: {num_sampled} samples per prompt (T={temperature})")
            merged_qa, merged_prompts = expand_response_samples(qa_pairs, prompts, num_sampled)
            all_answers, rich_traces = generate_fn(prompts=merged_prompts, temperature=temperature, **common_kwargs)
    else:
        if num_sampled > 1:
            print(f"[DLM] temperature=0: ignoring num_response_samples={num_sampled}, generating 1 greedy response per prompt")
        print("[DLM] Greedy-only: 1 deterministic response per prompt")
        merged_qa, merged_prompts = list(qa_pairs), list(prompts)
        all_answers, rich_traces = generate_fn(prompts=merged_prompts, temperature=0.0, **common_kwargs)
        config["num_response_samples"] = 1

    return merged_qa, merged_prompts, all_answers, rich_traces

def _interleave_answers(
    greedy_answers: list[str],
    sampled_answers: list[str],
    num_questions: int,
    num_sampled: int,
) -> list[str]:
    merged = []
    for q in range(num_questions):
        merged.append(greedy_answers[q])
        merged.extend(sampled_answers[q * num_sampled:(q + 1) * num_sampled])
    return merged


def _write_answers_jsonl(
    output_dir: str,
    qa_pairs: list[dict],
    prompts: list[str],
    answers: list[str],
    config_temperature: float,
) -> None:
    path = os.path.join(output_dir, "answers.jsonl")
    with open(path, "w") as f:
        for qa, prompt, answer in zip(qa_pairs, prompts, answers):
            mode = qa.get("generation_mode") if isinstance(qa, dict) else None
            if mode == "greedy" or config_temperature == 0:
                t = 0.0
            else:
                t = config_temperature
            question = qa.get("question", "") if isinstance(qa, dict) else ""
            reference_answer = qa.get("reference_answer", "") if isinstance(qa, dict) else ""
            record = {
                "prompt": prompt,
                "question": question,
                "reference_answer": reference_answer,
                "temperature": t,
                "answer": answer,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Answers: {path} ({len(answers)} entries)")

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config) # Retrieve config from file
    _apply_config(config, args) # Apply CLI overrides and set defaults in config

    # Set random seed for reproducibility
    seed = int(config["seed"])
    seed_everything(seed)

    # Set run_id and output directory for logging and saving results
    run_id = str(config.get("run_id") or derive_run_id(config))
    output_dir = args.output_dir or config.get("output_dir") or os.path.join("outputs", run_id)
    config["run_id"] = run_id

    # Select device
    try:
        device = select_device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DLM] Using device: {device}")

    # Load model, tokenizer, and prepare dataset inputs (prompts)
    backend = config["model_backend"]
    print(f"[DLM] Backend: {backend}, Model: {config['model_id']}")
    model, tokenizer, eos_token_ids, ptid = _setup_model(config, device)
    dataset_key, qa_pairs, prompts = _load_or_prepare_inputs(config, args, tokenizer, output_dir)

    # Common kwargs for generation function (both greedy and sampled)
    common_kwargs = dict(
        model=model, device=device, backend=backend,
        batch_size=config["batch_size"],
        tokenizer=tokenizer, steps=config["steps"], gen_length=config["gen_length"],
        remasking=resolve_remasking(config["remasking"]),
        mask_id=config["mask_id"], eos_token_ids=eos_token_ids,
        logits_eos_inf=config["logits_eos_inf"],
        confidence_eos_eot_inf=config["confidence_eos_eot_inf"],
        save_trajectory=True, k_topk_logits=config["topk_trace_k"],
    )

    # Run generation and collect traces
    merged_qa, merged_prompts, all_answers, rich_traces = _run_generation(
        generate_fn, qa_pairs, prompts, config, common_kwargs, seed,
    )

    # Save complete trace collection to .npz file and metadata to metadata.json
    summary = save_trace_collection(
        output_dir=output_dir, run_config=config, qa_pairs=merged_qa,
        prompts=merged_prompts, rich_traces=rich_traces, tokenizer=tokenizer,
        dataset_key=dataset_key, eos_token_ids=eos_token_ids, pad_token_id=ptid,
    )

    # Save the generated answers to answers.jsonl for easy reference
    _write_answers_jsonl(output_dir, merged_qa, merged_prompts, all_answers, float(config["temperature"]))

    # Update metadata.json to include reference to answers.jsonl
    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata.setdefault("files", {})["answers"] = "answers.jsonl"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"Examples: {summary['num_examples']}")

if __name__ == "__main__":
    main()
