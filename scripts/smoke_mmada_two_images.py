"""Smoke test: can MMaDA handle two images in a single prompt?

Builds the input as:
  [<|mmu|>, <|soi|>, img1_tokens, <|eoi|>, <|soi|>, img2_tokens, <|eoi|>, text_ids]

and runs a short generation to see if the model produces a coherent answer.

Usage:
  python scripts/smoke_mmada_two_images.py [--steps 64] [--max-gen-length 128]
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image, ImageDraw

from src.config import MMADA_VQ_MODEL_ID, MODEL_HF_IDS, Model
from src.generate.mmada_mmu import (
    encode_image_tokens as _encode_image,
    image_transform_squash as _image_transform,
)
from src.generate.model import (
    MMADA_MASK_ID,
    MMADA_SPECIAL_TOKENS,
    load_model,
    load_tokenizer,
    load_vq_model,
    infer_eos_token_ids,
)


def make_red_square(size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(255, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((size // 4, size // 2), "RED", fill=(255, 255, 255))
    return img


def make_blue_circle(size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], fill=(0, 0, 255))
    return img


def build_two_image_input(
    img1: Image.Image,
    img2: Image.Image,
    prompt_text: str,
    *,
    tokenizer,
    vq_model,
    device,
    mask_id: int = MMADA_MASK_ID,
) -> torch.Tensor:
    """Build MMaDA input with two images:
    [<|mmu|>, <|soi|>, img1_tokens, <|eoi|>, <|soi|>, img2_tokens, <|eoi|>, text_ids]
    """
    img1_tokens = _encode_image(img1, vq_model, tokenizer, device)  # (1, N)
    img2_tokens = _encode_image(img2, vq_model, tokenizer, device)  # (1, N)

    text_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    sp = MMADA_SPECIAL_TOKENS
    mmu = torch.tensor([[sp["<|mmu|>"]]], dtype=torch.long, device=device)
    soi = torch.tensor([[sp["<|soi|>"]]], dtype=torch.long, device=device)
    eoi = torch.tensor([[sp["<|eoi|>"]]], dtype=torch.long, device=device)

    input_ids = torch.cat([
        mmu,
        soi, img1_tokens, eoi,
        soi, img2_tokens, eoi,
        text_ids,
    ], dim=1)

    return input_ids


@torch.no_grad()
def generate_two_image(
    model,
    input_ids: torch.Tensor,
    *,
    tokenizer,
    steps: int = 128,
    max_gen_length: int = 128,
    mask_id: int = MMADA_MASK_ID,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
) -> str:
    """Generate using the model's built-in mmu_generate method."""
    prompt_len = input_ids.shape[1]

    x = model.mmu_generate(
        idx=input_ids,
        max_new_tokens=max_gen_length,
        steps=steps,
        block_length=max_gen_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
    )

    response_ids = x[:, prompt_len:]
    answer = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser(description="Smoke test: MMaDA with 2 images")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max-gen-length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = MODEL_HF_IDS[Model.MMaDA]
    print(f"Loading model: {model_id}")
    model = load_model(model_id, device)
    tokenizer = load_tokenizer(model_id)

    print(f"Loading VQ model: {MMADA_VQ_MODEL_ID}")
    vq_model = load_vq_model(MMADA_VQ_MODEL_ID, device)

    img1 = make_red_square()
    img2 = make_blue_circle()
    prompt = (
        "You are given two images. "
        "The first image is Image A and the second image is Image B. "
        "Please describe what you see in each image."
    )

    print(f"\nPrompt: {prompt}")
    print("Image 1: red square with white text")
    print("Image 2: white background with blue circle")

    print("\nBuilding two-image input...")
    input_ids = build_two_image_input(
        img1, img2, prompt,
        tokenizer=tokenizer, vq_model=vq_model, device=device,
    )
    print(f"Input sequence length: {input_ids.shape[1]}")

    print(f"\nRunning generation (steps={args.steps}, max_gen_length={args.max_gen_length})...")
    answer = generate_two_image(
        model, input_ids,
        tokenizer=tokenizer,
        steps=args.steps,
        max_gen_length=args.max_gen_length,
    )

    print(f"\n{'='*60}")
    print(f"MODEL ANSWER:\n{answer}")
    print(f"{'='*60}")

    if answer:
        print("\nSUCCESS: Model produced output with two-image input.")
    else:
        print("\nWARNING: Model produced empty output.")


if __name__ == "__main__":
    main()
