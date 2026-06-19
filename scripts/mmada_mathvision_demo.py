"""Visual smoke test: run MMaDA on a few MathVision questions.

Loads the first N MathVision examples (image + question), runs MMaDA mmu
generation, and prints the question / reference answer / model answer side by
side. The image actually fed to the model (squashed to 512x512, matching
MMaDA's image_transform_squash for diagram-type inputs) is saved next to the
original so accuracy can be inspected visually.

Usage (inside the container, see scripts/mmada_mathvision_demo.slurm):
  python scripts/mmada_mathvision_demo.py --num 5
  python scripts/mmada_mathvision_demo.py --num 5 \
      --parquet /lustre/work/pdl16831/usx56od/Datasets/MathVision/test-00000-of-00001-3532b8d3f1b4047a.parquet
"""

from __future__ import annotations

import argparse
import os

import torch

from src.config import MMADA_VQ_MODEL_ID, MODEL_HF_IDS, Model
from src.datasets.dataset_specific import mathvision
from src.datasets.dataloader import build_raw_prompts
from src.generate.mmada_mmu import generate_mmu
from src.generate.model import (
    MMADA_IMAGE_RESOLUTION,
    infer_eos_token_ids,
    load_model,
    load_tokenizer,
    load_vq_model,
)
from src.utils.io import write_jsonl


def load_samples(args):
    """Return (qa_samples, raw_prompts, images) for the first N examples."""
    if args.parquet:
        from datasets import Image, load_dataset

        ds = load_dataset("parquet", data_files=args.parquet, split="train")
        ds = ds.cast_column("decoded_image", Image())
        n = min(len(ds), args.num)
        samples = [mathvision.extract_qa_sample(ds[i]) for i in range(n)]
        samples = [s for s in samples if s is not None]
        prompts = [mathvision.format_prompt(s) for s in samples]
    else:
        gen_config = {
            "dataset": "mathvision",
            "num_questions": args.num,
            "fewshot_k": 0,
        }
        hf_token = os.environ.get("HF_TOKEN") or None
        _, samples, prompts = build_raw_prompts(gen_config, hf_token=hf_token)

    images = [s["image"] for s in samples]
    return samples, prompts, images


def save_images(samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    res = MMADA_IMAGE_RESOLUTION
    from PIL import Image

    for i, s in enumerate(samples):
        img = s["image"]
        stem = os.path.join(out_dir, f"ex{i:02d}")
        img.save(f"{stem}_original.png")
        # What MMaDA actually sees: bicubic squash to res x res (no crop).
        img.resize((res, res), Image.BICUBIC).save(f"{stem}_model_input.png")


def main():
    parser = argparse.ArgumentParser(description="MMaDA x MathVision visual smoke test")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--parquet", type=str, default=None,
                        help="Load directly from a local parquet file (offline).")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--vq-model-id", type=str, default=MMADA_VQ_MODEL_ID)
    parser.add_argument("--max-gen-length", type=int, default=1024,
                        help="Reasoning traces are long; MixCoT needs room before "
                             "the post-</think> answer. Try 1024-2048.")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"],
                        help="Token remasking strategy during denoising.")
    parser.add_argument("--confidence-eos-eot-inf",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="LLaDA-style EOS suppression (ON by default): zero EOS/EOT "
                             "confidence in the low_confidence remasker so it is never "
                             "committed early. Disable with --no-confidence-eos-eot-inf. "
                             "Requires eos_token_ids (inferred automatically).")
    parser.add_argument("--logits-eos-inf", 
                        help="Aggressive: set EOS/EOT logits to -inf before argmax so "
                             "they can never be predicted at all. Can cause run-on output.")
    parser.add_argument("--out-dir", type=str, default="mmada_mathvision_demo_out")
    parser.add_argument("--out-jsonl", type=str, default=None,
                        help="Path for the results JSONL (default: <out-dir>/results.jsonl).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = args.model_id or MODEL_HF_IDS[Model.MMaDA]

    print(f"Device:    {device}")
    print(f"Model:     {model_id}")
    print(f"VQ model:  {args.vq_model_id}")
    print(f"Gen:       length={args.max_gen_length} steps={args.steps} "
          f"block={args.block_length} temp={args.temperature}")
    print(f"Remask:    {args.remasking} "
          f"confidence_eos_eot_inf={args.confidence_eos_eot_inf} "
          f"logits_eos_inf={args.logits_eos_inf}")

    samples, prompts, images = load_samples(args)
    print(f"Loaded {len(samples)} MathVision examples")
    save_images(samples, args.out_dir)
    print(f"Saved original + model-input images to {args.out_dir}/")

    model = load_model(model_id, device)
    tokenizer = load_tokenizer(model_id)
    vq_model = load_vq_model(args.vq_model_id, device)

    # The exact text MMaDA sees after chat-template formatting (generate_mmu
    # applies this same template internally, with tokenize=True). The full model
    # input additionally prepends <|mmu|><|soi|> + 1024 image tokens + <|eoi|>.
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]

    eos_token_ids = infer_eos_token_ids(tokenizer, model_id)
    print(f"EOS ids:   {eos_token_ids}")

    answers, _ = generate_mmu(
        model, prompts, images, device,
        tokenizer=tokenizer, vq_model=vq_model,
        steps=args.steps, max_gen_length=args.max_gen_length,
        block_size=args.block_length, temperature=args.temperature,
        batch_size=args.batch_size,
        remasking=args.remasking,
        eos_token_ids=eos_token_ids,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        logits_eos_inf=args.logits_eos_inf,
    )

    out_jsonl = args.out_jsonl or os.path.join(args.out_dir, "results.jsonl")
    records = []
    for i, (s, ans) in enumerate(zip(samples, answers)):
        opts = s.get("options") or []
        question = mathvision._strip_image_tags(s.get("question", ""))
        parsed = mathvision.parse_answer(ans)

        print("\n" + "=" * 70)
        print(f"EXAMPLE {i}  id={s.get('id')}  subject={s.get('subject')}  level={s.get('level')}")
        print(f"image: original size={s['image'].size}")
        print("-" * 70)
        print(f"Q: {question}")
        if opts:
            for j, opt in enumerate(opts):
                print(f"   ({chr(ord('A') + j)}) {opt}")
        print("-" * 70)
        print("FORMATTED PROMPT (chat template, image tokens prepended at runtime):")
        print(formatted_prompts[i])
        print("-" * 70)
        print(f"REFERENCE: {s.get('reference_answer')}")
        print(f"MMaDA:     {ans}")
        print(f"PARSED:    {parsed}")

        records.append({
            "index":             i,
            "id":                s.get("id"),
            "subject":           s.get("subject"),
            "level":             s.get("level"),
            "question":          question,
            "raw_question":      s.get("question", ""),
            "options":           opts,
            "formatted_prompt":  formatted_prompts[i],
            "reference_answer":  s.get("reference_answer"),
            "solution":          s.get("raw_reference_answer", ""),
            "model_answer":      ans,
            "parsed_answer":     parsed,
            "image_size":        list(s["image"].size),
            "image_original":    os.path.join(args.out_dir, f"ex{i:02d}_original.png"),
            "image_model_input": os.path.join(args.out_dir, f"ex{i:02d}_model_input.png"),
        })

    write_jsonl(out_jsonl, records)
    print(f"\nWrote {len(records)} records to {out_jsonl}")


if __name__ == "__main__":
    main()
