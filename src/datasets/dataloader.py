"""Dataset dataloader: prompt building, chat templates, and sample expansion.

Unified interface between datasets and the generation pipeline.
"""

from __future__ import annotations

import inspect
from typing import Any

from src.datasets.loading import load_hf_dataset, load_local_dataset
from src.registry import get_dataset_module
from src.utils.io import read_jsonl, write_jsonl


_RECORD_ONLY_KEYS = {"prompt_id", "dataset", "prompt_text"}


def build_raw_prompt(dataset_module, qa_item, prefix=None):
    """Build the raw prompt text for one QA item (no chat template)."""
    question = qa_item.get("question", "") if isinstance(qa_item, dict) else qa_item[0]
    choices = qa_item.get("choices", []) if isinstance(qa_item, dict) else []

    sig = inspect.signature(dataset_module.format_prompt)
    kwargs = dict(prefix=prefix, choices=choices)
    for param_name in sig.parameters:
        if param_name in ("question", "prefix", "choices", "self"):
            continue
        if isinstance(qa_item, dict) and param_name in qa_item:
            kwargs[param_name] = qa_item[param_name]
    return dataset_module.format_prompt(question, **kwargs)


def build_fewshot_prefix(dataset_module, fewshot_pairs):
    """Build a few-shot prefix from the dataset's few_shot_prefix function."""
    if not fewshot_pairs or not hasattr(dataset_module, "few_shot_prefix"):
        return None
    examples = []
    for item in fewshot_pairs:
        question = item.get("question", "") if isinstance(item, dict) else item[0]
        ref = item.get("reference_answer", "") if isinstance(item, dict) else item[1]
        fewshot_answer = item.get("fewshot_answer", ref) if isinstance(item, dict) else ref
        example = {"question": question, "answer": fewshot_answer}
        choices = item.get("choices", []) if isinstance(item, dict) else []
        if choices:
            example["choices"] = choices
        examples.append(example)
    return dataset_module.few_shot_prefix(examples)


def build_prompt_records(
    generation_config: dict[str, Any],
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    """Load dataset and build prompt records (model-independent)."""
    dataset_key = generation_config["dataset"]
    dataset_module = get_dataset_module(dataset_key)

    split = generation_config.get("split", getattr(dataset_module, "DEFAULT_SPLIT", "validation"))
    config_name = generation_config.get("dataset_config_name", getattr(dataset_module, "DEFAULT_CONFIG_NAME", None))

    local_path = generation_config.get("local_dataset_path")
    if local_path:
        dataset = load_local_dataset(local_path)
    else:
        dataset = load_hf_dataset(
            dataset_module.HF_NAME,
            split=split,
            config_name=config_name,
            token=hf_token,
        )

    all_pairs = [
        p for i in range(len(dataset))
        if (p := dataset_module.extract_question_and_answer(dataset[i])) is not None
    ]

    fewshot_k = int(generation_config.get("fewshot_k", 0))
    fewshot_pairs = all_pairs[:fewshot_k]
    prefix = build_fewshot_prefix(dataset_module, fewshot_pairs)

    max_questions = int(generation_config.get("num_questions", 1000))
    qa_pairs = all_pairs[fewshot_k:fewshot_k + max_questions]

    records = []
    for i, item in enumerate(qa_pairs):
        prompt_text = build_raw_prompt(dataset_module, item, prefix=prefix)
        record = dict(item)
        record["prompt_id"] = i
        record["dataset"] = dataset_key
        record["prompt_text"] = prompt_text
        records.append(record)
    return records


def write_prompts_jsonl(records: list[dict[str, Any]], path: str) -> None:
    """Write prompt records to a JSONL file."""
    write_jsonl(path, records)
    print(f"Wrote {len(records)} prompts to {path}")


def apply_chat_templates(
    records: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict], list[str]]:
    """Extract QA pairs and apply chat template to prompt text."""
    qa_pairs = []
    prompts = []
    for rec in records:
        qa_pairs.append({k: v for k, v in rec.items() if k not in _RECORD_ONLY_KEYS})
        messages = [{"role": "user", "content": rec["prompt_text"]}]
        prompts.append(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        )
    return qa_pairs, prompts


def load_prompts_jsonl(
    path: str,
    tokenizer: Any,
) -> tuple[str, list[dict], list[str]]:
    """Load prompts.jsonl and apply chat template."""
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"No prompt records found in {path}")
    dataset_key = records[0]["dataset"]
    qa_pairs, prompts = apply_chat_templates(records, tokenizer)
    return dataset_key, qa_pairs, prompts


def prepare_dataset_inputs(
    generation_config: dict[str, Any],
    tokenizer: Any,
    hf_token: str | None,
) -> tuple[str, list[dict], list[str]]:
    """Load dataset, extract QA pairs, and build chat-templated prompts."""
    records = build_prompt_records(generation_config, hf_token)
    if not records:
        raise ValueError("No QA pairs found in dataset")
    dataset_key = records[0]["dataset"]
    qa_pairs, prompts = apply_chat_templates(records, tokenizer)
    return dataset_key, qa_pairs, prompts


def expand_greedy_and_sampled(
    qa_pairs: list[dict],
    prompts: list[str],
    num_sampled: int,
) -> tuple[list[dict], list[str]]:
    """Build merged QA pairs: per prompt, index 0 is greedy, indices 1..N are sampled."""
    total_per_prompt = 1 + num_sampled
    expanded_qa = []
    expanded_prompts = []
    for qa_index, (qa_item, prompt) in enumerate(zip(qa_pairs, prompts)):
        expanded_qa.append(_with_sample_fields(qa_item, qa_index, 0, total_per_prompt, "greedy"))
        expanded_prompts.append(prompt)
        for sid in range(1, total_per_prompt):
            expanded_qa.append(_with_sample_fields(qa_item, qa_index, sid, total_per_prompt, "sampled"))
            expanded_prompts.append(prompt)
    return expanded_qa, expanded_prompts


def expand_response_samples(
    qa_pairs: list[dict],
    prompts: list[str],
    num_response_samples: int,
) -> tuple[list[dict], list[str]]:
    """Repeat each QA/prompt pair for N response samples."""
    if num_response_samples == 1:
        return list(qa_pairs), list(prompts)
    expanded_qa = []
    expanded_prompts = []
    for qa_index, (qa_item, prompt) in enumerate(zip(qa_pairs, prompts)):
        for sid in range(num_response_samples):
            expanded_qa.append(_with_sample_fields(qa_item, qa_index, sid, num_response_samples))
            expanded_prompts.append(prompt)
    return expanded_qa, expanded_prompts


def interleave_traces(
    greedy_traces: dict[str, Any],
    sampled_traces: dict[str, Any],
    num_questions: int,
    num_sampled: int,
) -> dict[str, Any]:
    """Interleave greedy [Q,T,L] and sampled [Q*N,T,L] into [Q*(1+N),T,L]."""
    import torch
    interleaved = {}
    for key in sorted(greedy_traces):
        gt = greedy_traces[key]
        st = sampled_traces[key]
        chunks = []
        for q in range(num_questions):
            chunks.append(gt[q:q + 1])
            chunks.append(st[q * num_sampled:(q + 1) * num_sampled])
        interleaved[key] = torch.cat(chunks, dim=0)
    return interleaved


def _with_sample_fields(qa_item, qa_index, response_sample_id, num_response_samples, generation_mode="sampled"):
    """Attach row-level sample fields to a QA item."""
    example_id = _example_id(qa_item, qa_index)
    sample_id = f"{example_id}::sample_{response_sample_id}" if num_response_samples > 1 else str(example_id)
    extra = {
        "qa_index": int(qa_index),
        "response_sample_id": int(response_sample_id),
        "num_response_samples": int(num_response_samples),
        "qa_example_id": str(example_id),
        "sample_example_id": sample_id,
        "generation_mode": str(generation_mode),
    }
    if isinstance(qa_item, dict):
        item = dict(qa_item)
        item.update(extra)
        return item
    question = qa_item[0] if len(qa_item) > 0 else ""
    ref = qa_item[1] if len(qa_item) > 1 else ""
    return {"question": question, "reference_answer": ref, **extra}


def _example_id(qa_item, sample_id):
    """Extract or build an example ID from a QA item."""
    if isinstance(qa_item, dict):
        for key in ("sample_example_id", "id", "example_id", "question_id"):
            if key in qa_item and str(qa_item[key]).strip():
                return str(qa_item[key])
    return str(sample_id)
