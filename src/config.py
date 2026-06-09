"""Single source of truth for model, dataset, and remasking definitions.

All canonical names, HuggingFace IDs, backend types, and dataset defaults
live here. No aliases -- use the exact enum values everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Model(str, Enum):
    """Supported diffusion language models."""
    LLaDA = "LLaDA"
    LLaDA15 = "LLaDA1.5"
    Dream = "Dream"
    Nemotron = "Nemotron"


class Dataset(str, Enum):
    """Supported evaluation datasets."""
    triviaqa = "triviaqa"
    gsm8k = "gsm8k"
    wmt14_fr_en = "wmt14_fr_en"
    xsum = "xsum"
    samsum = "samsum"
    hotpotqa = "hotpotqa"
    musique = "musique"


class Remasking(str, Enum):
    """Remasking strategies for the denoising loop."""
    lc = "lc"
    rd = "rd"


MODEL_HF_IDS: dict[Model, str] = {
    Model.LLaDA: "GSAI-ML/LLaDA-8B-Instruct",
    Model.LLaDA15: "GSAI-ML/LLaDA-1.5",
    Model.Dream: "Dream-org/Dream-v0-Instruct-7B",
    Model.Nemotron: "nvidia/Nemotron-Labs-Diffusion-8B",
}

MODEL_BACKENDS: dict[Model, str] = {
    Model.LLaDA: "llada",
    Model.LLaDA15: "llada",
    Model.Dream: "dream",
    Model.Nemotron: "nemotron",
}

REMASKING_FULL_NAMES: dict[Remasking, str] = {
    Remasking.lc: "low_confidence",
    Remasking.rd: "random",
}


@dataclass(frozen=True)
class DatasetDefaults:
    """Default configuration for a dataset."""
    hf_name: str
    config_name: str | None
    split: str
    batch_size: int
    default_label_method: str
    label_gpu_mode: str
    fewshot_k: int = 0


def _defaults_from_adapter(dataset: Dataset, batch_size: int = 16) -> DatasetDefaults:
    """Derive DatasetDefaults from the adapter module's constants."""
    from src.registry import get_dataset_module
    mod = get_dataset_module(dataset.value)
    return DatasetDefaults(
        hf_name=getattr(mod, "HF_NAME"),
        config_name=getattr(mod, "DEFAULT_CONFIG_NAME", None),
        split=getattr(mod, "DEFAULT_SPLIT", "test"),
        batch_size=batch_size,
        default_label_method=getattr(mod, "DEFAULT_LABEL_METHOD", "llm_judge"),
        label_gpu_mode=getattr(mod, "LABEL_GPU_MODE", "gpu"),
        fewshot_k=getattr(mod, "FEWSHOT_K", 0),
    )


_DATASET_BATCH_SIZES: dict[Dataset, int] = {
    Dataset.triviaqa: 32,
    Dataset.gsm8k: 16,
    Dataset.wmt14_fr_en: 16,
    Dataset.xsum: 16,
    Dataset.samsum: 16,
    Dataset.hotpotqa: 16,
    Dataset.musique: 16,
}

_DATASET_DEFAULTS_CACHE: dict[Dataset, DatasetDefaults] | None = None


def _get_dataset_defaults() -> dict[Dataset, DatasetDefaults]:
    global _DATASET_DEFAULTS_CACHE
    if _DATASET_DEFAULTS_CACHE is None:
        _DATASET_DEFAULTS_CACHE = {
            ds: _defaults_from_adapter(ds, _DATASET_BATCH_SIZES.get(ds, 16))
            for ds in Dataset
        }
    return _DATASET_DEFAULTS_CACHE


class _DatasetDefaultsProxy:
    """Dict-like proxy that lazily builds DATASET_DEFAULTS on first access."""

    def __getitem__(self, key: Dataset) -> DatasetDefaults:
        return _get_dataset_defaults()[key]

    def __contains__(self, key: object) -> bool:
        return key in _get_dataset_defaults()

    def __iter__(self):
        return iter(_get_dataset_defaults())

    def items(self):
        return _get_dataset_defaults().items()

    def values(self):
        return _get_dataset_defaults().values()

    def keys(self):
        return _get_dataset_defaults().keys()

    def get(self, key, default=None):
        return _get_dataset_defaults().get(key, default)


DATASET_DEFAULTS: Any = _DatasetDefaultsProxy()

def run_id(
    model: Model,
    dataset: Dataset,
    length: int,
    steps: int,
    remasking: Remasking,
    *,
    temperature: float | None = None,
    fewshot_k: int | None = None,
) -> str:
    """Build the canonical run identifier used for output directory naming."""
    rid = f"{model.value}_{dataset.value}_l{length}_s{steps}_{remasking.value}"
    if temperature is not None:
        safe_t = str(temperature).replace("+", "").replace("-", "m").replace(".", "p")
        rid = f"{rid}_t{safe_t}"
    if fewshot_k is not None and fewshot_k > 0:
        rid = f"{rid}_fs{fewshot_k}"
    return rid


def config_filename(
    model: Model,
    dataset: Dataset,
    length: int,
    steps: int,
    remasking: Remasking,
    *,
    fewshot_k: int | None = None,
    confidence_eos_eot_inf: bool = False,
) -> str:
    """Build the config JSON filename."""
    base = f"{model.value}_{dataset.value}_l{length}_s{steps}_{remasking.value}"
    if fewshot_k is not None and fewshot_k > 0:
        base = f"{base}_fs{fewshot_k}"
    if confidence_eos_eot_inf:
        base = f"{base}_ceot"
    return f"{base}.json"


def build_generation_config(
    model: Model,
    dataset: Dataset,
    length: int,
    steps: int,
    remasking: Remasking,
    *,
    num_questions: int = 1000,
    temperature: float = 1.0,
    num_response_samples: int = 20,
    generate_greedy: bool = True,
    topk_trace_k: int = 64,
    fewshot_k: int | None = None,
    confidence_eos_eot_inf: bool = False,
    cfg_scale: float = 0.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a complete generation config from canonical parameters."""
    ds = DATASET_DEFAULTS[dataset]
    resolved_fewshot_k = ds.fewshot_k if fewshot_k is None else fewshot_k
    config: dict[str, Any] = {
        "model_family": "DLM",
        "model_id": MODEL_HF_IDS[model],
        "model_backend": MODEL_BACKENDS[model],
        "dataset": dataset.value,
        "dataset_config_name": ds.config_name,
        "split": ds.split,
        "batch_size": ds.batch_size,
        "num_questions": num_questions,
        "gen_length": length,
        "steps": steps,
        "remasking": remasking.value,
        "temperature": temperature,
        "num_response_samples": num_response_samples,
        "generate_greedy": generate_greedy,
        "topk_trace_k": topk_trace_k,
        "fewshot_k": resolved_fewshot_k,
        "save_full_trace": True,
        "confidence_eos_eot_inf": confidence_eos_eot_inf,
        "cfg_scale": cfg_scale,
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
    return run_id(
        model_enum, dataset_enum,
        int(config["gen_length"]), int(config["steps"]),
        remasking_enum,
        fewshot_k=fewshot_k if fewshot_k > 0 else None,
    )


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
