"""Dataset dataloader: prompt building, chat templates, and sample expansion.

Public interface:
  build_raw_prompts()     -- load dataset, return (dataset_key, QASamples, raw_prompt_texts)
  apply_chat_template()   -- apply model chat template to raw prompt strings
  write_prompts_jsonl()   -- persist raw prompt records to JSONL
  load_prompts_jsonl()    -- restore (dataset_key, QASamples, raw_prompts) from JSONL
  prepare_dataset_inputs() -- convenience: build_raw_prompts + apply_chat_template in one call

The generation layer (generate.py) is responsible for calling apply_chat_template.
The dataset layer never touches the tokenizer.
"""

from __future__ import annotations

from typing import Any

from src.config import DATASET_CONFIGS, Dataset
from src.datasets.loading import load_hf_dataset, load_local_dataset
from src.datasets.qa_pair import QASample
from src.registry import get_dataset_module
from src.utils.io import read_jsonl, write_jsonl


_RECORD_ONLY_KEYS = {"prompt_id", "dataset", "prompt_text"}


# ---------------------------------------------------------------------------
# Core data-preparation API
# ---------------------------------------------------------------------------

def build_raw_prompts(
    generation_config: dict[str, Any],
    hf_token: str | None = None,
    verbose: bool = True,
) -> tuple[str, list[QASample], list[str]]:
    """Load a dataset and build raw text prompts (no chat template applied).

    Returns
    -------
    dataset_key : str
    qa_samples  : list[QASample]   one entry per test question
    raw_prompts : list[str]        parallel list of raw prompt texts
    """
    dataset_key = generation_config["dataset"]
    dataset_module = get_dataset_module(dataset_key)
    ds_config = DATASET_CONFIGS[Dataset(dataset_key)]

    split = generation_config.get("split", ds_config.split)
    config_name = generation_config.get("dataset_config_name", ds_config.config_name)

    if ds_config.local_path:
        if verbose:
            print(f"Loading dataset from local path: {ds_config.local_path}")
        dataset = load_local_dataset(ds_config.local_path)
    else:
        assert ds_config.hf_name, (
            f"Dataset '{dataset_key}' has no hf_name and no local_path in DATASET_CONFIGS"
        )
        dataset = load_hf_dataset(
            ds_config.hf_name,
            split=split,
            config_name=config_name,
            token=hf_token,
        )

    # ``extract_qa_sample`` returns a QASample, ``None`` (skip), or a list of
    # QASamples. The list form lets one source row fan out into several samples
    # (e.g. ViLP pairs one question with three images -> three samples).
    all_samples: list[QASample] = []
    for i in range(len(dataset)):
        result = dataset_module.extract_qa_sample(dataset[i])
        if result is None:
            continue
        if isinstance(result, list):
            all_samples.extend(result)
        else:
            all_samples.append(result)

    fewshot_k = int(generation_config.get("fewshot_k", ds_config.fewshot_k))
    fewshot_samples = all_samples[:fewshot_k]
    prefix = _build_fewshot_prefix(dataset_module, fewshot_samples)

    max_questions = int(generation_config.get("num_questions", 1000))
    test_samples = all_samples[fewshot_k : fewshot_k + max_questions]

    raw_prompts = [
        dataset_module.format_prompt(item, fewshot_prefix=prefix)
        for item in test_samples
    ]
    return dataset_key, test_samples, raw_prompts


def apply_chat_template(raw_prompts: list[str], tokenizer: Any) -> list[str]:
    """Apply the model's chat template to a list of raw prompt strings."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in raw_prompts
    ]


def prepare_dataset_inputs(
    generation_config: dict[str, Any],
    tokenizer: Any,
    hf_token: str | None = None,
) -> tuple[str, list[QASample], list[str]]:
    """Load dataset, build prompts, and apply chat template.

    Returns
    -------
    dataset_key   : str
    qa_samples    : list[QASample]
    chat_prompts  : list[str]   ready for the model tokenizer
    """
    dataset_key, qa_samples, raw_prompts = build_raw_prompts(generation_config, hf_token)
    if not qa_samples:
        raise ValueError("No QA pairs found in dataset")
    chat_prompts = apply_chat_template(raw_prompts, tokenizer)
    return dataset_key, qa_samples, chat_prompts


# ---------------------------------------------------------------------------
# Prompt JSONL persistence
# ---------------------------------------------------------------------------

def write_prompts_jsonl(
    path: str,
    dataset_key: str,
    qa_samples: list[QASample],
    raw_prompts: list[str],
) -> None:
    """Write prompt records (QASample + raw prompt text) to a JSONL file."""
    records = [
        {"prompt_id": i, "dataset": dataset_key, "prompt_text": p, **s}
        for i, (s, p) in enumerate(zip(qa_samples, raw_prompts))
    ]
    write_jsonl(path, records)
    print(f"Wrote {len(records)} prompts to {path}")


def load_prompts_jsonl(path: str) -> tuple[str, list[QASample], list[str]]:
    """Load a prompts.jsonl file.

    Returns
    -------
    dataset_key : str
    qa_samples  : list[QASample]   (record keys minus prompt_id / dataset / prompt_text)
    raw_prompts : list[str]        raw prompt texts — caller must apply chat template
    """
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"No prompt records found in {path}")
    dataset_key = records[0]["dataset"]
    qa_samples: list[QASample] = []
    raw_prompts: list[str] = []
    for rec in records:
        raw_prompts.append(rec["prompt_text"])
        qa_samples.append({k: v for k, v in rec.items() if k not in _RECORD_ONLY_KEYS})
    return dataset_key, qa_samples, raw_prompts


# ---------------------------------------------------------------------------
# Sample expansion helpers (used by generate.py)
# ---------------------------------------------------------------------------

def expand_greedy_and_sampled(
    qa_samples: list[QASample],
    prompts: list[str],
    num_sampled: int,
) -> tuple[list[dict], list[str]]:
    """Build merged list: per prompt, index 0 is greedy, indices 1..N are sampled."""
    total_per_prompt = 1 + num_sampled
    expanded_qa: list[dict] = []
    expanded_prompts: list[str] = []
    for qa_index, (qa_item, prompt) in enumerate(zip(qa_samples, prompts)):
        expanded_qa.append(_with_sample_fields(qa_item, qa_index, 0, total_per_prompt, "greedy"))
        expanded_prompts.append(prompt)
        for sid in range(1, total_per_prompt):
            expanded_qa.append(_with_sample_fields(qa_item, qa_index, sid, total_per_prompt, "sampled"))
            expanded_prompts.append(prompt)
    return expanded_qa, expanded_prompts


def expand_response_samples(
    qa_samples: list[QASample],
    prompts: list[str],
    num_response_samples: int,
) -> tuple[list[dict], list[str]]:
    """Repeat each QA/prompt pair for N independent response samples."""
    if num_response_samples == 1:
        return list(qa_samples), list(prompts)
    expanded_qa: list[dict] = []
    expanded_prompts: list[str] = []
    for qa_index, (qa_item, prompt) in enumerate(zip(qa_samples, prompts)):
        for sid in range(num_response_samples):
            expanded_qa.append(_with_sample_fields(qa_item, qa_index, sid, num_response_samples))
            expanded_prompts.append(prompt)
    return expanded_qa, expanded_prompts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_fewshot_prefix(dataset_module: Any, fewshot_samples: list[QASample]) -> str | None:
    if not fewshot_samples or not hasattr(dataset_module, "few_shot_prefix"):
        return None
    return dataset_module.few_shot_prefix(fewshot_samples)


def _with_sample_fields(
    qa_item: QASample,
    qa_index: int,
    response_sample_id: int,
    num_response_samples: int,
    generation_mode: str = "sampled",
) -> dict[str, Any]:
    """Attach generation-tracking fields to a QASample, returning a plain dict."""
    example_id = _example_id(qa_item, qa_index)
    sample_id = (
        f"{example_id}::sample_{response_sample_id}"
        if num_response_samples > 1
        else str(example_id)
    )
    extra = {
        "qa_index":              int(qa_index),
        "response_sample_id":    int(response_sample_id),
        "num_response_samples":  int(num_response_samples),
        "qa_example_id":         str(example_id),
        "sample_example_id":     sample_id,
        "generation_mode":       str(generation_mode),
    }
    if isinstance(qa_item, dict):
        item = dict(qa_item)
        item.update(extra)
        return item
    question = qa_item[0] if len(qa_item) > 0 else ""
    ref = qa_item[1] if len(qa_item) > 1 else ""
    return {"question": question, "reference_answer": ref, **extra}


def _example_id(qa_item: Any, sample_id: int) -> str:
    """Extract or synthesize a stable example ID from a QASample."""
    if isinstance(qa_item, dict):
        for key in ("sample_example_id", "id", "example_id", "question_id"):
            if key in qa_item and str(qa_item[key]).strip():
                return str(qa_item[key])
    return str(sample_id)
