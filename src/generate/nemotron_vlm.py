"""Dedicated Nemotron-Labs-Diffusion-VLM-8B (image + text) inference.

NVIDIA's Nemotron-Labs-Diffusion-VLM-8B is the vision variant of the Nemotron
block-diffusion LLM. Per its model card the recipe is:

  * preprocess (image + text) messages with the repo's own ``process_messages``
    (``from image_processing import process_messages``), which returns
    ``input_ids``, ``pixel_values`` and ``image_sizes``;
  * generate with the threshold-driven diffusion loop
    ``model.generate(prompt_ids, pixel_values=, image_sizes=, max_new_tokens=,
    steps=, block_length=, shift_logits=False, threshold=, eos_token_id=)``,
    which returns ``(out_ids, nfe)``.

``image_processing`` is an auxiliary file in the model repo (not auto-registered),
so we make the downloaded snapshot importable before using it -- mirroring how the
MMaDA path vendors its own ``models`` package.

CAREFUL-ATTENTION ITEMS (verify against the downloaded repo before a full run):
  * ``process_messages`` expects messages in the OpenAI ``image_url`` schema with
    a URL/path. The bench supplies an in-memory PIL image, so we write it to a
    temporary PNG and reference that path. If a future ``image_processing.py``
    accepts PIL directly, prefer that to avoid the round-trip.
  * ``transformers>=5.0`` plus pillow / requests / opencv-python are required ->
    run in the separate "nemotron" environment (requirements_nemotron.in).

One (prompt, image) pair is processed at a time (batch_size accepted but ignored):
variable-length image+text sequences and the threshold-driven per-sequence loop
make per-item the safe choice (matches the MMaDA mmu path).

A single post-hoc forward (with the same pixel_values/image_sizes) yields top-k
log-probs, returning the same ``(answers, topk_data)`` contract as the other
backends.

Public API:
  generate_vlm(model, prompts, images, device, *, tokenizer, model_id, ...)
      -> (answers, topk_data)
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.generate.nemotron_diffusion import stack_ragged_topk

_PROCESS_MESSAGES_CACHE: dict[str, Callable] = {}


def _load_process_messages(model_id: str) -> Callable:
    """Import ``process_messages`` from the model repo, caching per model id.

    The repo's python files are fetched into the HF cache (a no-op once the model
    has been downloaded) and that snapshot directory is put on ``sys.path`` so the
    bare ``from image_processing import process_messages`` resolves.
    """
    if model_id in _PROCESS_MESSAGES_CACHE:
        return _PROCESS_MESSAGES_CACHE[model_id]

    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download(model_id, allow_patterns=["*.py"])
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from image_processing import process_messages  # type: ignore

    _PROCESS_MESSAGES_CACHE[model_id] = process_messages
    return process_messages


def _logits_of(output: Any) -> torch.Tensor:
    """Return the logits tensor from a forward pass (ModelOutput or raw tensor)."""
    return getattr(output, "logits", output)


@torch.no_grad()
def generate_vlm(
    model: Any,
    prompts: list[str],
    images: list[Image.Image],
    device: Any,
    *,
    tokenizer: Any,
    model_id: str,
    steps: int = 512,
    max_gen_length: int = 512,
    block_size: int | None = None,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int | None = None,
    eos_token_ids: list[int] | tuple[int, ...] | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    batch_size: int = 1,
    top_k: int = 64,
    # Nemotron-native knobs.
    threshold: float = 0.9,
    shift_logits: bool = False,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Generate answers for (image + text) prompts using Nemotron-Diffusion-VLM.

    ``temperature``/``cfg_scale``/``remasking``/``mask_id`` and the EOS-logit knobs
    are accepted for interface parity but inert (threshold-driven diffusion).
    Returns the same ``(answers, topk_data)`` contract.
    """
    assert len(prompts) == len(images), (
        f"prompts ({len(prompts)}) and images ({len(images)}) must have equal length"
    )
    process_messages = _load_process_messages(model_id)
    block_length = int(block_size) if block_size else 32
    eos_id = (
        eos_token_ids[0]
        if eos_token_ids
        else getattr(tokenizer, "eos_token_id", None)
    )
    is_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    all_answers: list[str] = []
    topk_logprobs_batches: list[torch.Tensor] = []
    topk_ids_batches: list[torch.Tensor] = []

    for prompt_text, image in tqdm(
        zip(prompts, images), total=len(prompts), desc="Generating (nemotron-vlm)",
    ):
        if image.mode != "RGB":
            image = image.convert("RGB")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.save(tmp.name)
            tmp.close()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": tmp.name}},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            batch = process_messages(tokenizer, messages, add_generation_prompt=True)
        finally:
            os.unlink(tmp.name)

        prompt_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        image_sizes = batch["image_sizes"]
        prompt_len = prompt_ids.shape[1]

        out = model.generate(
            prompt_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_new_tokens=max_gen_length,
            steps=steps,
            block_length=block_length,
            shift_logits=shift_logits,
            threshold=threshold,
            eos_token_id=eos_id,
        )
        out_ids = out[0] if isinstance(out, (tuple, list)) else out
        response_ids = out_ids[:, prompt_len:]

        final_logits = _logits_of(
            model(out_ids, pixel_values=pixel_values, image_sizes=image_sizes)
        )[:, prompt_len:, :].float()
        log_probs = F.log_softmax(final_logits, dim=-1)
        del final_logits
        k = min(top_k, log_probs.shape[-1])
        tk_logprobs, tk_ids = torch.topk(log_probs, k=k, dim=-1)
        del log_probs
        topk_logprobs_batches.append(tk_logprobs.cpu().to(torch.float32))
        topk_ids_batches.append(tk_ids.cpu().to(torch.int32))

        all_answers.append(
            tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
        )
        del out, out_ids, response_ids, pixel_values, tk_logprobs, tk_ids
        if is_cuda:
            torch.cuda.empty_cache()

    # Per-(image,prompt) answer spans differ in length -> pad to a common length
    # before stacking (same ragged-length issue as the text backend).
    topk_data = {
        "topk_logprobs": stack_ragged_topk(topk_logprobs_batches, 0.0),
        "topk_token_ids": stack_ragged_topk(topk_ids_batches, 0.0),
    }
    return all_answers, topk_data
