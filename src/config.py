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
    wmt14_de_en = "wmt14_de_en"
    xsum = "xsum"
    samsum = "samsum"


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


DATASET_DEFAULTS: dict[Dataset, DatasetDefaults] = {
    Dataset.triviaqa: DatasetDefaults(
        hf_name="mandarjoshi/trivia_qa",
        config_name="rc",
        split="validation",
        batch_size=32,
        default_label_method="llm_judge",
        label_gpu_mode="gpu",
        fewshot_k=0,
    ),
    Dataset.gsm8k: DatasetDefaults(
        hf_name="openai/gsm8k",
        config_name="main",
        split="test",
        batch_size=16,
        default_label_method="exact_match",
        label_gpu_mode="cpu",
        fewshot_k=4,
    ),
    Dataset.wmt14_fr_en: DatasetDefaults(
        hf_name="wmt/wmt14",
        config_name="fr-en",
        split="test",
        batch_size=16,
        default_label_method="llm_judge",
        label_gpu_mode="gpu",
        fewshot_k=0,
    ),
    Dataset.wmt14_de_en: DatasetDefaults(
        hf_name="wmt/wmt14",
        config_name="de-en",
        split="test",
        batch_size=16,
        default_label_method="llm_judge",
        label_gpu_mode="gpu",
        fewshot_k=0,
    ),
    Dataset.xsum: DatasetDefaults(
        hf_name="EdinburghNLP/xsum",
        config_name=None,
        split="test",
        batch_size=16,
        default_label_method="llm_judge",
        label_gpu_mode="gpu",
        fewshot_k=0,
    ),
    Dataset.samsum: DatasetDefaults(
        hf_name="knkarthick/samsum",
        config_name=None,
        split="test",
        batch_size=16,
        default_label_method="llm_judge",
        label_gpu_mode="gpu",
        fewshot_k=0,
    ),
}

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
