"""Smoke test: MMaDA with 1, 2, 3, 4, and 5 solid-color images.

Input format for N images:
  [<|mmu|>, <|soi|>, img1, <|eoi|>, ..., <|soi|>, imgN, <|eoi|>, text_ids]

Usage:
  python scripts/smoke_mmada_one_image.py --model-id Gen-Verse/MMaDA-8B-Base
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from PIL import Image

from src.config import MMADA_VQ_MODEL_ID, MODEL_HF_IDS, Model
from src.generate.mmada_mmu import encode_image_tokens as _encode_image
from src.generate.model import (
    MMADA_MASK_ID,
    MMADA_SPECIAL_TOKENS,
    load_model,
    load_tokenizer,
    load_vq_model,
)


def make_solid(color: tuple[int, int, int], size: int = 256) -> Image.Image:
    return Image.new("RGB", (size, size), color=color)


def build_multi_image_input(images, prompt_text, *, tokenizer, vq_model, device):
    """Build MMaDA input for 1..N images:
    [<|mmu|>, <|soi|>, img1, <|eoi|>, ..., <|soi|>, imgN, <|eoi|>, text_ids]
    """
    sp = MMADA_SPECIAL_TOKENS
    mmu = torch.tensor([[sp["<|mmu|>"]]], dtype=torch.long, device=device)
    soi = torch.tensor([[sp["<|soi|>"]]], dtype=torch.long, device=device)
    eoi = torch.tensor([[sp["<|eoi|>"]]], dtype=torch.long, device=device)

    parts = [mmu]
    for img in images:
        parts.extend([soi, _encode_image(img, vq_model, tokenizer, device), eoi])

    text_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)
    parts.append(text_ids)

    return torch.cat(parts, dim=1)


@torch.no_grad()
def generate(model, input_ids, *, tokenizer, steps, max_gen_length, block_length,
             temperature, mask_id=MMADA_MASK_ID, remasking="low_confidence"):
    prompt_len = input_ids.shape[1]
    x = model.mmu_generate(
        idx=input_ids, max_new_tokens=max_gen_length, steps=steps,
        block_length=block_length, temperature=temperature,
        remasking=remasking, mask_id=mask_id,
    )
    response_ids = x[:, prompt_len:]
    answer = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
    return answer, response_ids[0]


def print_debug(raw_ids, tokenizer):
    ids = raw_ids.cpu().tolist()
    n_mask = sum(1 for t in ids if t == MMADA_MASK_ID)
    n_special = sum(1 for t in ids if t >= 126080)
    n_text = sum(1 for t in ids if t < 126080)
    print(f"  tokens: {len(ids)} total, {n_text} text, {n_special} special, {n_mask} mask")
    print(f"  unique IDs: {len(set(ids))}")
    print(f"  top-10: {Counter(ids).most_common(10)}")
    print(f"  first 30 IDs: {ids[:30]}")
    raw = tokenizer.decode(raw_ids, skip_special_tokens=False)
    print(f"  raw (200 chars): {raw[:200]}")


def run_test(label, input_ids, *, model, tokenizer, args):
    num_blocks = args.max_gen_length // args.block_length
    steps_per_block = args.steps // num_blocks
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    print(f"Input length: {input_ids.shape[1]}")
    print(f"Config: steps={args.steps}, gen={args.max_gen_length}, "
          f"block={args.block_length} ({num_blocks}x{steps_per_block}), temp={args.temperature}")

    answer, raw_ids = generate(
        model, input_ids, tokenizer=tokenizer,
        steps=args.steps, max_gen_length=args.max_gen_length,
        block_length=args.block_length, temperature=args.temperature,
    )
    print_debug(raw_ids, tokenizer)
    print(f"\n  ANSWER: {answer}\n")
    return answer


def main():
    parser = argparse.ArgumentParser(description="Smoke test: MMaDA with solid-color images")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max-gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = args.model_id or MODEL_HF_IDS[Model.MMaDA]
    print(f"Device: {device}")
    print(f"Model: {model_id}")

    model = load_model(model_id, device)
    tokenizer = load_tokenizer(model_id)
    vq_model = load_vq_model(MMADA_VQ_MODEL_ID, device)

    colors = {
        "RED":    (255, 0, 0),
        "BLUE":   (0, 0, 255),
        "GREEN":  (0, 255, 0),
        "YELLOW": (255, 255, 0),
        "WHITE":  (255, 255, 255),
    }
    imgs = {name: make_solid(rgb) for name, rgb in colors.items()}
    kw = dict(tokenizer=tokenizer, vq_model=vq_model, device=device)

    # --- Test 1: single RED ---
    run_test("1 image — RED",
             build_multi_image_input([imgs["RED"]], "What color is this image?", **kw),
             model=model, tokenizer=tokenizer, args=args)

    # --- Test 2: single BLUE ---
    run_test("1 image — BLUE",
             build_multi_image_input([imgs["BLUE"]], "What color is this image?", **kw),
             model=model, tokenizer=tokenizer, args=args)

    # --- Test 3: 2 images ---
    run_test("2 images — RED, BLUE",
             build_multi_image_input(
                 [imgs["RED"], imgs["BLUE"]],
                 "You see two images. What color is the first image and what color is the second?",
                 **kw),
             model=model, tokenizer=tokenizer, args=args)

    # --- Test 4: 3 images ---
    run_test("3 images — RED, GREEN, BLUE",
             build_multi_image_input(
                 [imgs["RED"], imgs["GREEN"], imgs["BLUE"]],
                 "You see three images. Describe the color of each image in order.",
                 **kw),
             model=model, tokenizer=tokenizer, args=args)

    # --- Test 5: 5 images ---
    run_test("5 images — RED, BLUE, GREEN, YELLOW, WHITE",
             build_multi_image_input(
                 [imgs["RED"], imgs["BLUE"], imgs["GREEN"], imgs["YELLOW"], imgs["WHITE"]],
                 "You see five images. Describe the color of each image in order.",
                 **kw),
             model=model, tokenizer=tokenizer, args=args)


if __name__ == "__main__":
    main()
