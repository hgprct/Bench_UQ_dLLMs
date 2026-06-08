"""Model and tokenizer loading for diffusion language models."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any
from packaging import version

import torch, transformers
from transformers import AutoModel, AutoTokenizer

KNOWN_LLADA_EOS_EOT_IDS = (126081, 126348)


def load_model(model_id: str, device: Any):
    """Load a HuggingFace model with dLLM-compatible settings."""
    _validate_transformers_compat(model_id)
    with _quiet_dream_warning(model_id):
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    return model.to(device).eval()


def load_tokenizer(model_id: str):
    """Load a HuggingFace tokenizer with dLLM-compatible settings."""
    _validate_transformers_compat(model_id)
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _validate_transformers_compat(model_id: str) -> None:
    """Fail fast on known version incompatibilities."""
    tv = version.parse(transformers.__version__)
    mid = str(model_id).lower()
    if "llada" in mid and tv.major >= 5:
        raise RuntimeError(
            f"Incompatible transformers version {transformers.__version__} for {model_id}. "
            "LLaDA requires transformers 4.x. Install: pip install 'transformers>=4.48,<5'"
        )
    if "nemotron" in mid and tv.major < 5:
        raise RuntimeError(
            f"Incompatible transformers version {transformers.__version__} for {model_id}. "
            "Nemotron requires transformers 5.x. Install: pip install 'transformers>=5.0'"
        )

@contextmanager
def _quiet_dream_warning(model_id: str):
    """Silence HuggingFace's generic warning for Dream sampling args."""
    if "dream" not in str(model_id).lower():
        yield
        return
    hf_logging = getattr(getattr(transformers, "utils", None), "logging", None)
    prev = None
    if hf_logging and hasattr(hf_logging, "get_verbosity"):
        prev = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"The following generation flags are not valid.*")
            yield
    finally:
        if prev is not None and hasattr(hf_logging, "set_verbosity"):
            hf_logging.set_verbosity(prev)


def infer_eos_token_ids(tokenizer: Any, model_id: str | None = None, configured_ids: Any = None) -> list[int]:
    """Collect EOS/EOT ids from tokenizer + known LLaDA ids."""
    eos_ids = set(_as_int_list(configured_ids))
    eos_ids.update(_as_int_list(getattr(tokenizer, "eos_token_id", None)))
    eos_ids.update(_as_int_list(getattr(tokenizer, "eos_token_ids", None)))
    eos_ids.update(_as_int_list(getattr(tokenizer, "eot_token_id", None)))

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        for token in ("<|eot_id|>", "<|eot|>", "<|im_end|>"):
            token_id = convert(token)
            if token_id is not None and token_id != unk_id:
                eos_ids.add(int(token_id))

    if model_id is not None and "llada" in str(model_id).lower():
        eos_ids.update(KNOWN_LLADA_EOS_EOT_IDS)
    return sorted(eos_ids)


def infer_mask_token_id(tokenizer: Any, model: Any = None, configured_id: Any = None) -> int | None:
    """Infer the mask token ID from tokenizer, model, or config."""
    if configured_id is not None:
        return int(configured_id)
    for owner in (tokenizer, getattr(model, "generation_config", None), getattr(model, "config", None)):
        if owner is None:
            continue
        token_id = getattr(owner, "mask_token_id", None)
        if token_id is not None:
            return int(token_id)
    return None


def pad_token_id(tokenizer: Any) -> int | None:
    """Get pad token ID from tokenizer."""
    pid = getattr(tokenizer, "pad_token_id", None)
    return None if pid is None else int(pid)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value if item is not None]
    return [int(value)]
