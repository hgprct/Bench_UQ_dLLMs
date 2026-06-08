"""CLI: per-feature wall-clock latency profiler for UQ methods.

Independent of the AUROC/PRR evaluation pipeline. Outputs:
  <output_dir>/latency_summary.csv   per-feature mean/std over prompts
  <output_dir>/latency_raw.jsonl     one record per (prompt, feature)
  <output_dir>/latency_metadata.json run-level diagnostics

Generation timing: three independent measurements per prompt:
  1. Greedy (T=0, batch=1)
  2. Single stochastic (T>0, batch=1)
  3. Batched stochastic (T>0, batch=N)

The baseline cost is token-based UQ (≈ greedy generation time). The overhead
column in the summary is the additional cost above this baseline.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from src.config import load_config
from src.io.json_utils import write_json, write_jsonl
from src.seed import seed_everything


SUMMARY_COLUMNS = [
    "run_id", "model_id", "dataset", "family", "feature",
    "n_prompts", "n_extra_gen", "n_nli_pairs",
    "time_gen_mean", "time_gen_std",
    "time_nli_mean", "time_nli_std",
    "time_math_mean", "time_math_std",
    "time_total_mean", "time_total_std",
    "overhead_mean", "overhead_std",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile per-feature UQ latency.")
    p.add_argument("--config", required=True, help="Path to generation config JSON")
    p.add_argument("--output_dir", required=True, help="Directory to write latency outputs")
    p.add_argument("--num_prompts", type=int, default=10)
    p.add_argument("--warmup_prompts", type=int, default=2)
    p.add_argument("--selection_dir", default=None,
                   help="Directory holding trained_weights.npz for selected-* features")
    p.add_argument("--budget_k", type=int, default=20,
                   help="K for top-K subset of QP weights (selected-*)")
    p.add_argument("--full_trajectory", action="store_true",
                   help="Profile full-trajectory features (O(T^2) NLI pairs; slow)")
    p.add_argument("--max_nli_batch", type=int, default=None,
                   help="Override autodetected NLI batch size")
    p.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import torch

    from src.generate._device import select_device
    from src.generate.model import (
        infer_eos_token_ids,
        infer_mask_token_id,
        load_model,
        load_tokenizer,
        pad_token_id as get_pad_token_id,
    )
    from src.semantic.entailment import load_entailment_model
    from src.timing.clock import autodetect_nli_batch, warmup_nli
    from src.timing import generate as gen_mod
    from src.timing.runner import (
        RunDiagnostics, aggregate, load_qp_weights, profile_one_prompt, sample_prompts,
    )

    args = parse_args(argv)
    seed_everything(args.seed)

    config = load_config(args.config)
    config.setdefault("batch_size", 1)
    config.setdefault("seed", args.seed)
    config.setdefault("model_backend", "llada")
    config.setdefault("topk_trace_k", 64)
    config.setdefault("logits_eos_inf", False)
    config.setdefault("confidence_eos_eot_inf", False)

    device = select_device()
    print(f"[latency] Device: {device}")

    model_id = config["model_id"]
    dataset = config.get("dataset", "unknown")
    print(f"[latency] Loading dLLM: {model_id}")
    model = load_model(model_id, device)
    tokenizer = load_tokenizer(model_id)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    eos_token_ids = infer_eos_token_ids(tokenizer, model_id=model_id)
    if get_pad_token_id(tokenizer) is None and eos_token_ids:
        tokenizer.pad_token_id = eos_token_ids[0]
    if config.get("mask_id") is None:
        config["mask_id"] = infer_mask_token_id(tokenizer, model=model)
    if config.get("mask_id") is None:
        raise ValueError("mask_id could not be inferred; set it in the config")
    config["eos_token_ids"] = eos_token_ids

    print(f"[latency] Loading NLI: {args.nli_model}")
    nli_model = load_entailment_model(args.nli_model, device=str(device))

    if args.max_nli_batch is not None:
        nli_batch = int(args.max_nli_batch)
        print(f"[latency] NLI batch (override): {nli_batch}")
    else:
        nli_batch = autodetect_nli_batch(nli_model, device)
        print(f"[latency] NLI batch (autodetected): {nli_batch}")
    warmup_nli(nli_model, device, batch_size=nli_batch)

    selected_steps = load_qp_weights(args.selection_dir, budget_k=args.budget_k)
    if selected_steps is None:
        print(f"[latency] No QP weights at {args.selection_dir}; skipping selected-* features")
    else:
        print(f"[latency] Selected step indices (K={len(selected_steps)}): {selected_steps}")

    n_response_samples = int(config.get("num_response_samples", 20))
    if args.full_trajectory:
        print(f"[latency] Full-trajectory features enabled (O(T^2) NLI pairs)")
    else:
        print(f"[latency] Full-trajectory features disabled (use --full_trajectory to enable)")
    print(f"[latency] Stochastic samples per prompt: {n_response_samples} (batched)")

    hf_token = os.environ.get("HF_TOKEN", "")
    total_needed = args.warmup_prompts + args.num_prompts
    all_prompts = sample_prompts(config, tokenizer, num_prompts=total_needed,
                                 hf_token=hf_token, seed=args.seed)
    if len(all_prompts) < total_needed:
        raise ValueError(f"Dataset yielded {len(all_prompts)} prompts; need {total_needed}")

    warmup_prompts = all_prompts[:args.warmup_prompts]
    timed_prompts = all_prompts[args.warmup_prompts:]

    print(f"[latency] Warmup: {len(warmup_prompts)} prompts (untimed)")
    for p in warmup_prompts:
        gen_mod.time_greedy_one(model, p, config=config, tokenizer=tokenizer,
                                eos_token_ids=eos_token_ids, device=device)

    print(f"[latency] Profiling {len(timed_prompts)} prompts")
    all_records = []
    greedy_times = []
    single_stoch_times = []
    batched_stoch_times = []
    for i, prompt in enumerate(timed_prompts):
        recs, gen_timings = profile_one_prompt(
            i, prompt,
            model=model, tokenizer=tokenizer, nli_model=nli_model,
            config=config, eos_token_ids=eos_token_ids, device=device,
            selected_steps=selected_steps, nli_batch_size=nli_batch,
            full_trajectory=args.full_trajectory,
        )
        all_records.extend(recs)
        greedy_times.append(gen_timings.greedy)
        single_stoch_times.append(gen_timings.single_stochastic)
        batched_stoch_times.append(gen_timings.batched_stochastic)
        print(
            f"[latency] prompt {i+1}/{len(timed_prompts)}: "
            f"greedy={gen_timings.greedy:.2f}s, "
            f"single_stoch={gen_timings.single_stochastic:.2f}s, "
            f"batched_stoch({gen_timings.n_samples})={gen_timings.batched_stochastic:.2f}s, "
            f"features={len(recs)}"
        )
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    import numpy as np
    n_timed = len(timed_prompts)
    ddof = 1 if n_timed > 1 else 0
    diagnostics = RunDiagnostics(
        n_prompts=n_timed,
        n_warmup=len(warmup_prompts),
        n_response_samples=n_response_samples,
        steps=int(config["steps"]),
        gen_length=int(config["gen_length"]),
        budget_k=args.budget_k if selected_steps else None,
        selected_step_indices=selected_steps,
        nli_batch_size=nli_batch,
        greedy_gen_time_mean=float(np.mean(greedy_times)),
        greedy_gen_time_std=float(np.std(greedy_times, ddof=ddof)),
        single_stochastic_gen_time_mean=float(np.mean(single_stoch_times)),
        single_stochastic_gen_time_std=float(np.std(single_stoch_times, ddof=ddof)),
        batched_stochastic_gen_time_mean=float(np.mean(batched_stoch_times)),
        batched_stochastic_gen_time_std=float(np.std(batched_stoch_times, ddof=ddof)),
    )

    run_id = config.get("run_id") or Path(args.config).stem
    summary = aggregate(all_records)

    # Baseline = mean token-feature total time (≈ greedy generation)
    token_totals = [s["time_total_mean"] for (fam, _), s in summary.items() if fam == "token"]
    baseline = float(np.mean(token_totals)) if token_totals else diagnostics.greedy_gen_time_mean

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "latency_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for (family, feature), stats in summary.items():
            overhead = max(0.0, stats["time_total_mean"] - baseline)
            row = {
                "run_id": run_id, "model_id": model_id, "dataset": dataset,
                "family": family, "feature": feature,
                "overhead_mean": overhead,
                "overhead_std": stats["time_total_std"],
                **stats,
            }
            writer.writerow(row)
    print(f"[latency] Wrote {summary_path}")

    raw_path = out_dir / "latency_raw.jsonl"
    write_jsonl(raw_path, ({
        "run_id": run_id,
        "model_id": model_id,
        "dataset": dataset,
        "prompt_idx": r.prompt_idx,
        "family": r.family,
        "feature": r.feature,
        "n_extra_gen": r.n_extra_gen,
        "n_nli_pairs": r.n_nli_pairs,
        "time_gen": r.time_gen,
        "time_nli": r.time_nli,
        "time_math": r.time_math,
        "time_total": r.time_total,
    } for r in all_records))
    print(f"[latency] Wrote {raw_path}")

    meta_path = out_dir / "latency_metadata.json"
    write_json(meta_path, {
        "run_id": run_id,
        "config_path": str(args.config),
        "model_id": model_id,
        "dataset": dataset,
        "steps": diagnostics.steps,
        "gen_length": diagnostics.gen_length,
        "remasking": config.get("remasking"),
        "temperature": config.get("temperature"),
        "num_response_samples": diagnostics.n_response_samples,
        "budget_k": diagnostics.budget_k,
        "selected_step_indices": diagnostics.selected_step_indices,
        "nli_model": args.nli_model,
        "nli_batch_size": diagnostics.nli_batch_size,
        "n_prompts": diagnostics.n_prompts,
        "n_warmup": diagnostics.n_warmup,
        "full_trajectory": args.full_trajectory,
        "baseline_time": baseline,
        "generation_timings": {
            "greedy_mean": diagnostics.greedy_gen_time_mean,
            "greedy_std": diagnostics.greedy_gen_time_std,
            "single_stochastic_mean": diagnostics.single_stochastic_gen_time_mean,
            "single_stochastic_std": diagnostics.single_stochastic_gen_time_std,
            "batched_stochastic_mean": diagnostics.batched_stochastic_gen_time_mean,
            "batched_stochastic_std": diagnostics.batched_stochastic_gen_time_std,
            "n_stochastic_samples": n_response_samples,
        },
    })
    print(f"[latency] Wrote {meta_path}")


if __name__ == "__main__":
    main()
