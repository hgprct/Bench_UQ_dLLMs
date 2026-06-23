"""Model-side preparation of generation inputs from the dataset's output.

Architecture boundary
---------------------
The dataset layer (``src.datasets``) is model-agnostic: it extracts rows, builds
``QASample`` objects, and renders each one into a plain-text **user message**
(dataset instruction + content) via ``format_prompt``. It never touches a
tokenizer and knows nothing about any model.

This module is the *model* side of that boundary. Given the dataset's output it
performs the model-specific prompt construction and the generation-time
fan-out:

  apply_chat_template()       -- wrap user messages in the model's chat template
  expand_response_samples()   -- repeat each prompt for N independent samples
  expand_greedy_and_sampled() -- per prompt: index 0 greedy, 1..N sampled

The multimodal backends (``mmada_mmu``, ``nemotron_vlm``) build their own
message structure around the image directly from the raw user messages, so they
do not use ``apply_chat_template``; the text backends consume its output.
"""

from __future__ import annotations

from typing import Any

from src.datasets.dataloader import build_raw_prompts
from src.datasets.qa_pair import QASample


# ---------------------------------------------------------------------------
# Chat-template construction (text backends)
# ---------------------------------------------------------------------------

def apply_chat_template(user_messages: list[str], tokenizer: Any) -> list[str]:
    """Wrap each model-agnostic user message in the model's chat template.

    This is model-specific prompt construction: it depends entirely on the
    tokenizer's ``chat_template`` and is therefore the generation layer's job,
    not the dataset's.
    """
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for message in user_messages
    ]


def prepare_dataset_inputs(
    generation_config: dict[str, Any],
    tokenizer: Any,
    hf_token: str | None = None,
) -> tuple[str, list[QASample], list[str]]:
    """Load a dataset and produce chat-templated prompts in one call.

    Convenience wrapper: ``build_raw_prompts`` (dataset layer) renders the
    model-agnostic user messages, then ``apply_chat_template`` (model layer)
    constructs the final text-model inputs. Used by the text backends only.

    Returns ``(dataset_key, qa_samples, chat_prompts)``.
    """
    dataset_key, qa_samples, user_messages = build_raw_prompts(generation_config, hf_token)
    if not qa_samples:
        raise ValueError("No QA pairs found in dataset")
    chat_prompts = apply_chat_template(user_messages, tokenizer)
    return dataset_key, qa_samples, chat_prompts


# ---------------------------------------------------------------------------
# Generation sample expansion
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
    # Fallback for non-dict items (legacy tuples); QASamples are always dicts.
    question = qa_item[0] if len(qa_item) > 0 else ""
    ref = qa_item[1] if len(qa_item) > 1 else ""
    return {"question": question, "ground_truth_answer": ref, **extra}


def _example_id(qa_item: Any, sample_id: int) -> str:
    """Extract or synthesize a stable example ID from a QASample."""
    if isinstance(qa_item, dict):
        for key in ("sample_example_id", "id", "example_id", "question_id"):
            if key in qa_item and str(qa_item[key]).strip():
                return str(qa_item[key])
    return str(sample_id)
