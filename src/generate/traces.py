"""Trace serialization: building JSONL records and saving NPZ/metadata."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

PARSER_VERSION = "qa_trace_v3"


def save_trace_collection(
    output_dir: str,
    run_config: dict[str, Any],
    qa_pairs: list[dict],
    prompts: list[str],
    rich_traces: dict[str, Any],
    tokenizer: Any,
    dataset_key: str,
    eos_token_ids: list[int],
    pad_token_id: int | None = None,
    parse_answer: Any = None,
) -> dict[str, Any]:
    """Write config.json, examples.jsonl, traces.npz, and metadata.json."""
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.json")
    examples_path = os.path.join(output_dir, "examples.jsonl")
    traces_path = os.path.join(output_dir, "traces.npz")

    # Write config first so the directory is never empty while the save phase runs
    config = _build_trace_config(run_config, dataset_key, 0, eos_token_ids, pad_token_id, output_dir)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)

    examples = build_trace_example_records(
        qa_pairs, prompts, rich_traces, tokenizer, dataset_key,
        parse_answer=parse_answer,
    )

    config["num_examples"] = len(examples)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)

    with open(examples_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=True, allow_nan=False) + "\n")

    trace_arrays = _to_numpy_traces(rich_traces, release_tensors=True)
    np.savez_compressed(traces_path, **trace_arrays)

    metadata = {
        "run_id": config.get("run_id", "unknown"),
        "num_samples": len(examples),
        "run_config": run_config,
        "files": {
            "config": "config.json",
            "examples": "examples.jsonl",
            "traces": "traces.npz",
        },
        "trace_collection": {
            "format": "npz",
            "parser_version": PARSER_VERSION,
            "arrays": sorted(trace_arrays),
        },
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            prev = json.load(f)
        prev.update(metadata)
        metadata = prev
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "config_path": config_path,
        "examples_path": examples_path,
        "traces_path": traces_path,
        "num_examples": len(examples),
        "topk_saved": "topk_logits" in trace_arrays,
    }


def build_trace_example_records(qa_pairs, prompts, rich_traces, tokenizer, dataset_key, parse_answer=None):
    """Build JSONL example records with decoded strings per step."""
    response_token_ids = rich_traces["response_token_ids"]
    x0_pred_token_ids = rich_traces["x0_pred_token_ids"]
    num_examples, num_steps, gen_length = response_token_ids.shape

    # Convert to numpy once — avoids millions of individual tensor ops
    if hasattr(response_token_ids, "numpy"):
        r_np = response_token_ids.detach().cpu().numpy()
        x0_np = x0_pred_token_ids.detach().cpu().numpy()
    else:
        r_np = np.asarray(response_token_ids)
        x0_np = np.asarray(x0_pred_token_ids)

    # Batch-decode all (example, step) pairs in two calls instead of 2*N*S serial calls
    r_flat = r_np.reshape(-1, gen_length).tolist()
    x0_flat = x0_np.reshape(-1, gen_length).tolist()
    r_decoded = tokenizer.batch_decode(r_flat, skip_special_tokens=True)
    x0_decoded = tokenizer.batch_decode(x0_flat, skip_special_tokens=True)

    records = []
    for sample_id, qa_item in enumerate(qa_pairs):
        base = sample_id * num_steps
        final_response = r_decoded[base + num_steps - 1].strip()

        steps = [
            {
                "step": step_i,
                "x_string": r_decoded[base + step_i].strip(),
                "x0_string": x0_decoded[base + step_i].strip(),
            }
            for step_i in range(num_steps)
        ]

        parsed = parse_answer(final_response) if parse_answer else None

        record = dict(qa_item) if isinstance(qa_item, dict) else {}
        if not record.get("aliases"):
            record["aliases"] = [record.get("reference_answer", "")]
        record.update({
            "example_id": _example_id(qa_item, sample_id),
            "sample_id": sample_id,
            "dataset": dataset_key,
            "raw_answer": final_response,
            "parsed_answer": parsed,
            "prompt": prompts[sample_id],
            "final": {"response": final_response},
            "steps": steps,
        })
        records.append(record)
    return records


def _example_id(qa_item, sample_id):
    from src.datasets.dataloader import _example_id as _shared_example_id
    return _shared_example_id(qa_item, sample_id)


def _to_numpy_traces(rich_traces, *, release_tensors=False):
    """Convert trace tensors to numpy arrays."""
    import torch
    arrays = {}
    for key in sorted(rich_traces):
        val = rich_traces[key]
        if isinstance(val, torch.Tensor):
            arrays[key] = val.detach().cpu().numpy()
            if release_tensors:
                del rich_traces[key]
        else:
            arr = np.asarray(val)
            if arr.dtype == object:
                raise ValueError(f"Trace '{key}' has object dtype")
            arrays[key] = arr
    return arrays


def _build_trace_config(run_config, dataset_key, num_examples, eos_token_ids, pad_token_id, output_dir):
    run_id = str(run_config.get("run_id", "unknown"))
    return {
        "run_id": run_id,
        "model_id": run_config.get("model_id"),
        "model_backend": run_config.get("model_backend", "llada"),
        "dataset": dataset_key,
        "split": run_config.get("split"),
        "label_method": run_config.get("label_method"),
        "num_examples": num_examples,
        "num_questions": int(run_config.get("num_questions", num_examples)),
        "num_response_samples": int(run_config.get("num_response_samples", 1)),
        "steps": int(run_config.get("steps", 0)),
        "gen_length": int(run_config.get("gen_length", 0)),
        "temperature": float(run_config.get("temperature", 0.0)),
        "generate_greedy": bool(run_config.get("generate_greedy", True)),
        "remasking": run_config.get("remasking"),
        "mask_id": int(run_config.get("mask_id", 126336)),
        "eos_token_ids": [int(t) for t in eos_token_ids],
        "pad_token_id": pad_token_id,
        "seed": int(run_config.get("seed", 42)),
        "parser_version": PARSER_VERSION,
        "output_dir": output_dir,
    }
