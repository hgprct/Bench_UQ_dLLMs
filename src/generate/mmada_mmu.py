"""Dedicated MMaDA multimodal (image + text) inference.

Rigorous re-use of MMaDA's *own* multimodal-understanding (mmu) recipe, recovered
from the true repo (``MMaDA/``):

  * generation is delegated verbatim to ``MMadaModelLM.mmu_generate`` (the official
    block-wise low-confidence denoising loop in
    ``MMaDA/models/modeling_mmada.py``);
  * images are tokenised with the MAGVIT-v2 VQ model (``vq_model.get_code``) and
    offset by ``len(tokenizer)`` exactly as in ``MMaDA/inference_mmu.py``;
  * the input layout is ``[<|mmu|>, <|soi|>, image_tokens, <|eoi|>, chat_text]``;
  * the squash/centre-crop image transforms mirror
    ``MMaDA/training/utils.py``.

On top of the official path we add a single post-hoc forward pass over the
completed sequence to extract top-k log-probabilities, so this function returns
the **same** ``(answers, topk_data)`` contract as the text-only
``src.generate.denoising.generate`` and plugs into the rest of the UQ pipeline
(traces, features, evaluation) unchanged.

One (prompt, image) pair is processed at a time (batch_size = 1), matching the
validated reference scripts: image+text sequences have variable length and the
official loop does not pad, so per-item generation is the rigorous choice.

Public API:
  generate_mmu(model, prompts, images, device, *, tokenizer, vq_model, ...)
      -> (answers, topk_data)
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.generate.model import (
    MMADA_IMAGE_RESOLUTION,
    MMADA_MASK_ID,
    MMADA_SPECIAL_TOKENS,
)

# --- Image preprocessing ---------------------------------------------------
# Faithful copies of MMaDA/training/utils.py: both normalise to [-1, 1] with
# mean/std = 0.5. ``image_transform_squash`` resizes straight to (res, res)
# without cropping -- the correct choice for diagrams/figures/OCR (MathVision,
# ai2d, docvqa, geo, ...) so equations and labels are never cropped away.
# ``image_transform`` (resize short side + centre crop) suits natural photos.

_NORMALIZE = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def image_transform_squash(
    image: Image.Image, resolution: int = MMADA_IMAGE_RESOLUTION
) -> torch.Tensor:
    image = transforms.Resize(
        (resolution, resolution),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )(image)
    image = transforms.ToTensor()(image)
    return _NORMALIZE(image)


def image_transform(
    image: Image.Image, resolution: int = MMADA_IMAGE_RESOLUTION
) -> torch.Tensor:
    image = transforms.Resize(
        resolution, interpolation=transforms.InterpolationMode.BICUBIC,
    )(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    return _NORMALIZE(image)


def encode_image_tokens(
    image: Image.Image,
    vq_model: Any,
    tokenizer: Any,
    device: Any,
    *,
    transform: Callable[[Image.Image], torch.Tensor] = image_transform_squash,
) -> torch.Tensor:
    """PIL image -> VQ token ids (shape (1, N)), offset by the text-vocab size."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    # Offset image codes past the text vocabulary, as in MMaDA/inference_mmu.py.
    return vq_model.get_code(img_tensor) + len(tokenizer)


def build_mmu_input(
    image_tokens: torch.Tensor,
    text_token_ids: torch.Tensor,
    device: Any,
) -> torch.Tensor:
    """Build ``[<|mmu|>, <|soi|>, image_tokens, <|eoi|>, text_ids]`` (batch 1)."""
    sp = MMADA_SPECIAL_TOKENS
    one = lambda tok: torch.full(  # noqa: E731
        (image_tokens.shape[0], 1), sp[tok], dtype=torch.long, device=device,
    )
    if text_token_ids.dim() == 1:
        text_token_ids = text_token_ids.unsqueeze(0)
    return torch.cat(
        [one("<|mmu|>"), one("<|soi|>"), image_tokens, one("<|eoi|>"),
         text_token_ids.to(device)],
        dim=1,
    ).long()


@torch.no_grad()
def generate_mmu(
    model: Any,
    prompts: list[str],
    images: list[Image.Image],
    device: Any,
    *,
    tokenizer: Any,
    vq_model: Any,
    steps: int = 256,
    max_gen_length: int = 512,
    block_size: int | None = None,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = MMADA_MASK_ID,
    use_squash: bool = True,
    top_k: int = 64,
    # Accepted for interface parity with denoising.generate(); the official
    # mmu_generate handles EOS natively, so these knobs are intentionally inert.
    eos_token_ids: list[int] | tuple[int, ...] | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    batch_size: int = 1,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Generate answers for multimodal (image + text) prompts using MMaDA.

    Returns the same ``(answers, topk_data)`` contract as
    ``src.generate.denoising.generate``.
    """
    if block_size is None:
        block_size = max_gen_length
    assert max_gen_length % block_size == 0, (
        f"max_gen_length ({max_gen_length}) must be divisible by block_size ({block_size})"
    )
    assert len(prompts) == len(images), (
        f"prompts ({len(prompts)}) and images ({len(images)}) must have equal length"
    )

    transform = image_transform_squash if use_squash else image_transform
    is_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    all_answers: list[str] = []
    topk_logprobs_batches: list[torch.Tensor] = []
    topk_ids_batches: list[torch.Tensor] = []

    for prompt_text, image in tqdm(
        zip(prompts, images), total=len(prompts), desc="Generating (mmu)",
    ):
        image_tokens = encode_image_tokens(
            image, vq_model, tokenizer, device, transform=transform,
        )
        text_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        input_ids = build_mmu_input(image_tokens, text_ids, device)
        prompt_len = input_ids.shape[1]

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if is_cuda else nullcontext()
        )
        with autocast:
            # Official MMaDA denoising loop, used verbatim.
            x = model.mmu_generate(
                input_ids,
                max_new_tokens=max_gen_length,
                steps=steps,
                block_length=block_size,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
            )
            # Single post-hoc forward pass over the completed sequence to read
            # off the answer-region top-k log-probs (mirrors the text path).
            final_logits = model(x).logits[:, prompt_len:, :]

        response_ids = x[:, prompt_len:]
        log_probs = F.log_softmax(final_logits.float(), dim=-1)
        del final_logits
        k = min(top_k, log_probs.shape[-1])
        tk_logprobs, tk_ids = torch.topk(log_probs, k=k, dim=-1)
        del log_probs
        topk_logprobs_batches.append(tk_logprobs.cpu().to(torch.float32))
        topk_ids_batches.append(tk_ids.cpu().to(torch.int32))

        all_answers.append(
            tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
        )
        del x, response_ids, tk_logprobs, tk_ids
        if is_cuda:
            torch.cuda.empty_cache()

    topk_data = {
        "topk_logprobs": torch.cat(topk_logprobs_batches, dim=0),
        "topk_token_ids": torch.cat(topk_ids_batches, dim=0),
    }
    return all_answers, topk_data
