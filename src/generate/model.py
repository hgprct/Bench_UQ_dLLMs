"""Model and tokenizer loading for LLaDA diffusion language models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

KNOWN_LLADA_EOS_EOT_IDS = (126081, 126348)

# --- MMaDA multimodal constants -------------------------------------------
# Reserved special-token ids (MMaDA/training/prompting_utils.py).
MMADA_SPECIAL_TOKENS: dict[str, int] = {
    "<|soi|>": 126084,
    "<|eoi|>": 126085,
    "<|sov|>": 126086,
    "<|eov|>": 126087,
    "<|t2i|>": 126088,
    "<|mmu|>": 126089,
    "<|t2v|>": 126090,
    "<|v2v|>": 126091,
    "<|lvg|>": 126092,
    "[iPAD]":  126093,
    "<|r2i|>": 126094,
}

# Diffusion mask id (MMaDA/models/modeling_mmada.py).
MMADA_MASK_ID: int = 126336

# Image resolution used by the mmu pipeline (configs/mmada_demo.yaml -> 512).
# A 512x512 image is encoded by MAGVIT-v2 into 32x32 = 1024 discrete tokens.
MMADA_IMAGE_RESOLUTION: int = 512

# Path to the MMaDA source checkout (the "true repo" at the project root). It
# provides the `models` package (MAGVITv2, MMadaModelLM) and the `training`
# package; the repo expects its own directory on sys.path so that its internal
# absolute imports (`from models import ...`) resolve.
_MMADA_VENDOR_DIR = Path(__file__).resolve().parents[2] / "MMaDA"


def _ensure_mmada_importable() -> None:
    import sys
    path = str(_MMADA_VENDOR_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_vq_model(vq_model_id: str, device: Any):
    """Load the MAGVIT-v2 VQ image tokenizer used by MMaDA mmu."""
    _ensure_mmada_importable()
    from models import MAGVITv2

    vq_model = MAGVITv2().from_pretrained(vq_model_id)
    vq_model = vq_model.to(device).eval()
    vq_model.requires_grad_(False)
    return vq_model


def load_model(model_id: str, device: Any):
    """Load the model for *model_id* using its family's loader.

    MMaDA needs its vendored ``MMadaModelLM`` class. Every other supported
    family -- LLaDA / LLaDA-1.5, Dream, and both Nemotron-Diffusion variants --
    exposes its modeling code via ``trust_remote_code`` and loads through the
    generic ``AutoModel`` path (Dream registers ``AutoModel`` in its config;
    Nemotron's card uses ``AutoModel`` too).
    """
    from src.config import model_family

    if model_family(model_id) == "mmada":
        _ensure_mmada_importable()
        from models import MMadaModelLM

        model = MMadaModelLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        return model.to(device).eval()

    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    return model.to(device).eval()


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def infer_eos_token_ids(
    tokenizer: Any,
    model_id: str | None = None,
    configured_ids: Any = None,
) -> list[int]:
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


def infer_mask_token_id(
    tokenizer: Any,
    model: Any = None,
    configured_id: Any = None,
) -> int | None:
    if configured_id is not None:
        return int(configured_id)
    for owner in (
        tokenizer,
        getattr(model, "generation_config", None),
        getattr(model, "config", None),
    ):
        if owner is None:
            continue
        token_id = getattr(owner, "mask_token_id", None)
        if token_id is not None:
            return int(token_id)
    return None


def pad_token_id(tokenizer: Any) -> int | None:
    pid = getattr(tokenizer, "pad_token_id", None)
    return None if pid is None else int(pid)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value if item is not None]
    return [int(value)]
