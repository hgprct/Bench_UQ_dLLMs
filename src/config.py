"""Single source of truth for all pipeline configuration.

All canonical names, HuggingFace IDs, backend types, dataset-specific settings,
and generation defaults live here. No aliases -- use exact enum values everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Model(str, Enum):
    """Supported diffusion language models."""
    LLaDA    = "LLaDA"
    LLaDA15  = "LLaDA1.5"


class Dataset(str, Enum):
    """Supported evaluation datasets."""
    triviaqa    = "triviaqa"
    gsm8k       = "gsm8k"
    wmt14_fr_en = "wmt14_fr_en"
    xsum        = "xsum"
    samsum      = "samsum"
    hotpotqa    = "hotpotqa"
    musique     = "musique"


class Remasking(str, Enum):
    """Remasking strategies for the denoising loop."""
    lc = "lc"
    rd = "rd"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_HF_IDS: dict[Model, str] = {
    Model.LLaDA:    "GSAI-ML/LLaDA-8B-Instruct",
    Model.LLaDA15:  "GSAI-ML/LLaDA-1.5",
}

REMASKING_FULL_NAMES: dict[Remasking, str] = {
    Remasking.lc: "low_confidence",
    Remasking.rd: "random",
}


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Complete configuration for one dataset."""
    hf_name:         str            # HuggingFace dataset identifier (empty string if loading locally)
    local_path:      str | None     # Local file path; takes precedence over hf_name when set
    config_name:     str | None     # HuggingFace dataset config variant (e.g. "rc", "fr-en")
    split:           str            # Default split to use (train / validation / test)
    label_method:    str            # "exact_match" or "llm_judge"
    label_gpu_mode:  str            # "cpu" or "gpu" — whether labeling requires a GPU-hosted judge
    fewshot_k:       int            # Default number of few-shot examples to prepend
    batch_size:      int            # Default generation batch size
    max_gen_length:  int            # Maximum tokens to generate
    steps:           int            # Default denoising steps
    block_size:      int | None     # Block-diffusion block size (None = no blocking, uses max_gen_length)


DATASET_CONFIGS: dict[Dataset, DatasetConfig] = {
    Dataset.gsm8k: DatasetConfig(
        hf_name         = "openai/gsm8k",
        local_path      = None,
        config_name     = "main",
        split           = "test",
        label_method    = "exact_match",
        label_gpu_mode  = "cpu",
        fewshot_k       = 0,
        batch_size      = 16,
        max_gen_length  = 256,
        steps           = 256,
        block_size      = 32,
    ),
    Dataset.triviaqa: DatasetConfig(
        hf_name         = "mandarjoshi/trivia_qa",
        local_path      = None,
        config_name     = "rc",
        split           = "validation",
        label_method    = "llm_judge",
        label_gpu_mode  = "gpu",
        fewshot_k       = 0,
        batch_size      = 32,
        max_gen_length  = 32,
        steps           = 32,
        block_size      = 32,
    ),
    Dataset.wmt14_fr_en: DatasetConfig(
        hf_name         = "wmt/wmt14",
        local_path      = None,
        config_name     = "fr-en",
        split           = "test",
        label_method    = "llm_judge",
        label_gpu_mode  = "gpu",
        fewshot_k       = 0,
        batch_size      = 16,
        max_gen_length  = 128,
        steps           = 128,
        block_size      = 32,
    ),
    Dataset.xsum: DatasetConfig(
        hf_name         = "EdinburghNLP/xsum",
        local_path      = None,
        config_name     = None,
        split           = "test",
        label_method    = "llm_judge",
        label_gpu_mode  = "gpu",
        fewshot_k       = 0,
        batch_size      = 16,
        max_gen_length  = 128,
        steps           = 128,
        block_size      = 32,
    ),
    Dataset.samsum: DatasetConfig(
        hf_name         = "knkarthick/samsum",
        local_path      = None,
        config_name     = None,
        split           = "test",
        label_method    = "llm_judge",
        label_gpu_mode  = "gpu",
        fewshot_k       = 0,
        batch_size      = 16,
        max_gen_length  = 128,
        steps           = 128,
        block_size      = 32,
    ),
    Dataset.hotpotqa: DatasetConfig(
        hf_name         = "hotpotqa/hotpot_qa",
        local_path      = None,
        config_name     = "fullwiki",
        split           = "validation",
        label_method    = "llm_judge",
        label_gpu_mode  = "gpu",
        fewshot_k       = 0,
        batch_size      = 16,
        max_gen_length  = 64,
        steps           = 64,
        block_size      = 32,
    ),
    Dataset.musique: DatasetConfig(
        hf_name         = "",
        local_path      = "musique_data_v1.0/data/musique_full_v1.0_dev.jsonl",
        config_name     = None,
        split           = "train",
        label_method    = "llm_judge",
        label_gpu_mode  = "gpu",
        fewshot_k       = 0,
        batch_size      = 16,
        max_gen_length  = 64,
        steps           = 64,
        block_size      = 32,
    ),
}


# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------

GENERATION_DEFAULTS: dict[str, Any] = {
    "model_id":               "GSAI-ML/LLaDA-8B-Instruct",
    "seed":                   42,
    "steps":                  128,
    "max_gen_length":         128,
    "block_size":             None,
    "temperature":            0.0,
    "cfg_scale":              0.0,
    "remasking":              "lc",
    "batch_size":             8,
    "top_k":                  64,
    "num_response_samples":   20,
    "generate_greedy":        True,
    "num_questions":          1000,
    "fewshot_k":              0,
    "logits_eos_inf":         False,
    "confidence_eos_eot_inf": False,
}


# ---------------------------------------------------------------------------
# Run ID / config filename helpers
# ---------------------------------------------------------------------------

def run_id(
    model: Model,
    dataset: Dataset,
    max_gen_length: int,
    steps: int,
    remasking: Remasking,
    *,
    block_size: int | None = None,
    temperature: float | None = None,
    fewshot_k: int | None = None,
) -> str:
    """Build the canonical run identifier used for output directory naming."""
    rid = f"{model.value}_{dataset.value}_l{max_gen_length}_s{steps}_{remasking.value}"
    if block_size is not None:
        rid = f"{rid}_b{block_size}"
    if temperature is not None:
        safe_t = str(temperature).replace("+", "").replace("-", "m").replace(".", "p")
        rid = f"{rid}_t{safe_t}"
    if fewshot_k is not None and fewshot_k > 0:
        rid = f"{rid}_fs{fewshot_k}"
    return rid


def config_filename(
    model: Model,
    dataset: Dataset,
    max_gen_length: int,
    steps: int,
    remasking: Remasking,
    *,
    block_size: int | None = None,
    fewshot_k: int | None = None,
    confidence_eos_eot_inf: bool = False,
) -> str:
    """Build the config JSON filename."""
    base = f"{model.value}_{dataset.value}_l{max_gen_length}_s{steps}_{remasking.value}"
    if block_size is not None:
        base = f"{base}_b{block_size}"
    if fewshot_k is not None and fewshot_k > 0:
        base = f"{base}_fs{fewshot_k}"
    if confidence_eos_eot_inf:
        base = f"{base}_ceot"
    return f"{base}.json"


def build_generation_config(
    model: Model,
    dataset: Dataset,
    max_gen_length: int | None = None,
    steps: int | None = None,
    remasking: Remasking = Remasking.lc,
    *,
    block_size: int | None = ...,
    num_questions: int = 1000,
    temperature: float = 1.0,
    num_response_samples: int = 20,
    generate_greedy: bool = True,
    top_k: int = 64,
    fewshot_k: int | None = None,
    confidence_eos_eot_inf: bool = False,
    cfg_scale: float = 0.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a complete generation config from canonical parameters.

    When *max_gen_length*, *steps*, or *block_size* is omitted the per-dataset
    default from ``DATASET_CONFIGS`` is used.  Pass ``block_size=None``
    explicitly to force no blocking regardless of the dataset default.
    """
    ds = DATASET_CONFIGS[dataset]
    resolved_fewshot_k = ds.fewshot_k if fewshot_k is None else fewshot_k
    resolved_length = ds.max_gen_length if max_gen_length is None else max_gen_length
    resolved_steps = ds.steps if steps is None else steps
    resolved_block_size = ds.block_size if block_size is ... else block_size
    config: dict[str, Any] = {
        "model_family":          "DLM",
        "model_id":              MODEL_HF_IDS[model],
        "dataset":               dataset.value,
        "dataset_config_name":   ds.config_name,
        "split":                 ds.split,
        "label_method":          ds.label_method,
        "batch_size":            ds.batch_size,
        "num_questions":         num_questions,
        "max_gen_length":        resolved_length,
        "steps":                 resolved_steps,
        "block_size":            resolved_block_size,
        "remasking":             remasking.value,
        "temperature":           temperature,
        "cfg_scale":             cfg_scale,
        "num_response_samples":  num_response_samples,
        "generate_greedy":       generate_greedy,
        "top_k":                 top_k,
        "fewshot_k":             resolved_fewshot_k,
        "confidence_eos_eot_inf": confidence_eos_eot_inf,
    }
    config.update(overrides)
    return config


def derive_run_id(config: dict[str, Any]) -> str:
    """Derive canonical run_id from a config dict when not explicitly set."""
    hf_id_to_model = {v: k for k, v in MODEL_HF_IDS.items()}
    model_enum = hf_id_to_model.get(config.get("model_id", ""))
    if model_enum is None:
        return "default"
    try:
        dataset_enum = Dataset(config["dataset"])
        remasking_enum = Remasking(config["remasking"])
    except (ValueError, KeyError):
        return "default"
    fewshot_k = int(config.get("fewshot_k") or 0)
    bs = config.get("block_size")
    rid = run_id(
        model_enum, dataset_enum,
        int(config["max_gen_length"]), int(config["steps"]),
        remasking_enum,
        block_size=int(bs) if bs is not None else None,
        fewshot_k=fewshot_k if fewshot_k > 0 else None,
    )
    return rid


def resolve_remasking(short_name: str) -> str:
    """Convert short remasking name ('lc', 'rd') to full name for the denoising loop."""
    try:
        return REMASKING_FULL_NAMES[Remasking(short_name)]
    except ValueError:
        return short_name


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file."""
    with open(path) as f:
        return json.load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a JSON config file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
