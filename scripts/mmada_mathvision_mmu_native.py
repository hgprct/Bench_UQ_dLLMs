"""Answer the first N MathVision questions using MMaDA's *own* inference code.

Unlike scripts/mmada_mathvision_demo.py (which calls this repo's
src/generate/mmada_mmu.py wrapper), this script reproduces the MMaDA reference
inference path as closely as possible by importing directly from the MMaDA
source checkout (MMaDA/):

  * models.MAGVITv2 / models.MMadaModelLM          (model + VQ tokenizer classes)
  * training.prompting_utils.UniversalPrompting     (special-token bookkeeping)
  * training.utils.image_transform[_squash]         (the exact image preprocessing)
  * MMadaModelLM.mmu_generate                        (the block-wise denoising loop)

The input layout, image-token offset and generation call all match
MMaDA/inference_mmu.py. The only project-specific piece is loading MathVision
(via src/datasets/dataset_specific/mathvision.py) and the MixCoT "thinking mode"
prompt prefix taken verbatim from MMaDA/app.py.

Run inside the container (see scripts/mmada_mathvision_mmu_native.slurm):
  python scripts/mmada_mathvision_mmu_native.py --num 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# --- make the vendored MMaDA repo importable -------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MMADA_ROOT = os.path.join(_PROJECT_ROOT, "MMaDA")
if _MMADA_ROOT not in sys.path:
    sys.path.insert(0, _MMADA_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# MMaDA's own code -- this is the inference machinery we want to reuse.
from models import MAGVITv2, MMadaModelLM  # noqa: E402
from training.prompting_utils import UniversalPrompting  # noqa: E402
from training.utils import image_transform, image_transform_squash  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# Project-side data loading only.
from src.datasets.dataset_specific import mathvision  # noqa: E402

MMU_MODEL_PATH = "Gen-Verse/MMaDA-8B-MixCoT"
VQ_MODEL_NAME = "showlab/magvitv2"
RESOLUTION = 512  # MAGVIT-v2 -> 32x32 = 1024 image tokens
MAX_TEXT_LEN = 512
MASK_ID = 126336

# Verbatim from MMaDA/app.py: triggers the MixCoT model's chain-of-thought.
THINKING_PREFIX = (
    "Answer the following question about the image, explaining your reasoning step by step."
)


def build_prompt(sample) -> str:
    """Question (+ multiple-choice options) wrapped in the MixCoT think prompt."""
    question = mathvision._strip_image_tags(sample.get("question", ""))
    options = sample.get("options") or []
    parts = [question]
    if options:
        lines = [f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)]
        parts.append("Options:\n" + "\n".join(lines))
    parts.append("Put your final answer in \\boxed{}.")
    return THINKING_PREFIX + "\n\n".join(parts)


def load_samples(parquet_path: str, num: int):
    """First `num` MathVision examples as project QASample dicts."""
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=parquet_path, split="train")
    ds = ds.cast_column("decoded_image", HFImage())
    n = min(len(ds), num)
    samples = [mathvision.extract_qa_sample(ds[i]) for i in range(n)]
    return [s for s in samples if s is not None]


def main():
    p = argparse.ArgumentParser(description="MMaDA-native MathVision inference")
    p.add_argument("--num", type=int, default=10)
    p.add_argument(
        "--parquet",
        type=str,
        default="/lustre/work/pdl16831/usx56od/Datasets/MathVision/"
        "test-00000-of-00001-3532b8d3f1b4047a.parquet",
    )
    p.add_argument("--model-path", type=str, default=MMU_MODEL_PATH)
    p.add_argument("--vq-model", type=str, default=VQ_MODEL_NAME)
    # Generation knobs. Defaults give reasoning room while satisfying MMaDA's
    # constraints: gen_length % block_length == 0 and steps % num_blocks == 0.
    p.add_argument("--gen-length", type=int, default=1024)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--block-length", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--remasking", type=str, default="low_confidence",
                   choices=["low_confidence", "random"])
    p.add_argument("--out", type=str,
                   default="mmada_mathvision_demo_out/native_results.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model exactly like MMaDA/inference_mmu.py --------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=MAX_TEXT_LEN,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=0.1,
        use_reserved_token=True,
    )
    vq_model = MAGVITv2.from_pretrained(args.vq_model).to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)
    model = MMadaModelLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    image_offset = len(uni_prompting.text_tokenizer)
    print(f"Device={device}  model={args.model_path}")
    print(f"image_token_offset={image_offset}  mask_id={MASK_ID}")

    samples = load_samples(args.parquet, args.num)
    print(f"Loaded {len(samples)} MathVision examples")

    results = []
    for idx, sample in enumerate(samples):
        prompt_text = build_prompt(sample)
        image_ori = sample["image"]  # already RGB PIL

        # MathVision is diagram/figure-heavy -> squash (no crop), matching the
        # geo/ai2d/clevr/docvqa/llava branch in MMaDA/inference_mmu.py.
        image = image_transform_squash(image_ori, resolution=RESOLUTION).to(device)
        image = image.unsqueeze(0)
        image_tokens = vq_model.get_code(image) + image_offset

        messages = [{"role": "user", "content": prompt_text}]
        text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        bsz = image_tokens.shape[0]
        input_ids = torch.cat([
            (torch.ones(bsz, 1) * uni_prompting.sptids_dict["<|mmu|>"]).to(device),
            (torch.ones(bsz, 1) * uni_prompting.sptids_dict["<|soi|>"]).to(device),
            image_tokens,
            (torch.ones(bsz, 1) * uni_prompting.sptids_dict["<|eoi|>"]).to(device),
            text_token_ids,
        ], dim=1).long()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output_ids = model.mmu_generate(
                input_ids,
                max_new_tokens=args.gen_length,
                steps=args.steps,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                remasking=args.remasking,
                mask_id=MASK_ID,
            )

        generated_ids = output_ids[:, input_ids.shape[1]:]
        response = uni_prompting.text_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed = mathvision.parse_answer(response)
        reference = sample.get("reference_answer")

        print("\n" + "=" * 70)
        print(f"EXAMPLE {idx}  id={sample.get('id')}  subject={sample.get('subject')}")
        print(f"Q: {mathvision._strip_image_tags(sample.get('question',''))}")
        for j, opt in enumerate(sample.get("options") or []):
            print(f"   ({chr(ord('A') + j)}) {opt}")
        print(f"REFERENCE: {reference}")
        print(f"PARSED:    {parsed}")
        print(f"RAW:       {response}")

        results.append({
            "index": idx,
            "id": sample.get("id"),
            "subject": sample.get("subject"),
            "level": sample.get("level"),
            "question": mathvision._strip_image_tags(sample.get("question", "")),
            "options": sample.get("options") or [],
            "reference_answer": reference,
            "model_answer": response,
            "parsed_answer": parsed,
            "correct": parsed.strip() == str(reference).strip(),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n_correct = sum(r["correct"] for r in results)
    print(f"\nSaved {len(results)} results to {args.out}  "
          f"(exact-match {n_correct}/{len(results)})")


if __name__ == "__main__":
    main()
