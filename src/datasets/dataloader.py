"""Dataset dataloader: extraction, QA-object creation, and user-message rendering.

This module is the *dataset* side of the dataset/model boundary. It is fully
model-agnostic -- it never imports torch, never touches a tokenizer, and knows
nothing about any specific model. Its output (``QASample`` objects + rendered
plain-text user messages) is what every model consumes.

Public interface:
  build_raw_prompts()   -- load dataset, return (dataset_key, QASamples, user_messages)
  write_prompts_jsonl() -- persist QASamples + user messages to JSONL
  load_prompts_jsonl()  -- restore (dataset_key, QASamples, user_messages) from JSONL

Model-specific prompt construction (chat templating) and generation-time sample
expansion live in ``src.generate.inputs`` -- the model side of the boundary.
"""

from __future__ import annotations

from typing import Any

from src.config import DATASET_CONFIGS, Dataset
from src.datasets.loading import load_hf_dataset, load_local_dataset
from src.datasets.outputs import images_dir, save_images
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
    """Load a dataset and render model-agnostic user messages.

    Each user message is the dataset's instruction + content (built by the
    dataset adapter's ``format_prompt``); it carries no model-specific framing.
    The model layer turns these into model inputs (chat template / image tokens).

    Returns
    -------
    dataset_key   : str
    qa_samples    : list[QASample]   one entry per test question
    user_messages : list[str]        parallel list of rendered user messages
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

    # ``extract_qa_sample(row, id)`` returns a QASample, ``None`` (skip), or a list
    # of QASamples. The id is the row index; adapters with a native id override it.
    # The list form lets one source row fan out into several samples (e.g. ViLP
    # pairs one question with three images -> three samples).
    all_samples: list[QASample] = []
    for i in range(len(dataset)):
        result = dataset_module.extract_qa_sample(dataset[i], str(i))
        if result is None:
            continue
        if isinstance(result, list):
            all_samples.extend(result)
        else:
            all_samples.append(result)

    max_questions = int(generation_config.get("num_questions", 1000))
    test_samples = all_samples[:max_questions]

    # Persist any images (multimodal datasets) once to the shared per-dataset
    # folder so every model reuses the same files; text-only rows carry no image.
    n_imgs = save_images(dataset_key, test_samples)
    if verbose and n_imgs:
        print(f"Saved {n_imgs} new images to {images_dir(dataset_key)}")

    # full_prompt is precomputed by each adapter's extract_qa_sample; the
    # model-agnostic user message is exactly that text.
    user_messages = [s["full_prompt"] for s in test_samples]
    return dataset_key, test_samples, user_messages


# ---------------------------------------------------------------------------
# Prompt JSONL persistence
# ---------------------------------------------------------------------------

def write_prompts_jsonl(
    path: str,
    dataset_key: str,
    qa_samples: list[QASample],
    raw_prompts: list[str],
) -> None:
    """Write prompt records (QASample + user message text) to a JSONL file.

    The ``image`` field (a PIL.Image on multimodal datasets) is dropped: it is
    not JSON-serialisable, and the persisted prompts file is text-only (the
    multimodal backends re-decode images from the source dataset, not from here).
    """
    records = [
        {
            "prompt_id": i,
            "dataset": dataset_key,
            "prompt_text": p,
            **{k: v for k, v in s.items() if k != "image"},
        }
        for i, (s, p) in enumerate(zip(qa_samples, raw_prompts))
    ]
    write_jsonl(path, records)
    print(f"Wrote {len(records)} prompts to {path}")


def load_prompts_jsonl(path: str) -> tuple[str, list[QASample], list[str]]:
    """Load a prompts.jsonl file.

    Returns
    -------
    dataset_key   : str
    qa_samples    : list[QASample]   (record keys minus prompt_id / dataset / prompt_text)
    user_messages : list[str]        rendered user messages -- caller applies chat template
    """
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"No prompt records found in {path}")
    dataset_key = records[0]["dataset"]
    qa_samples: list[QASample] = []
    user_messages: list[str] = []
    for rec in records:
        user_messages.append(rec["prompt_text"])
        qa_samples.append({k: v for k, v in rec.items() if k not in _RECORD_ONLY_KEYS})
    return dataset_key, qa_samples, user_messages
