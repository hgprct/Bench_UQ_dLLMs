"""Visual smoke test: run google/diffusiongemma-26B-A4B-it on MathVision.

DiffusionGemma is a discrete-diffusion, MoE, multimodal Gemma-4 model. Unlike the
MMaDA path in this repo (which VQ-encodes a *squashed* 512x512 image into discrete
tokens), DiffusionGemma uses a native Gemma-4 vision encoder that handles variable
aspect ratio and resolution. So:

  * Do NOT squash / centre-crop the image -- pass the RGB PIL image as-is and let
    Gemma4ImageProcessor resize it. (do_normalize=False, just rescale to [0,1].)
  * Place the image BEFORE the text in the prompt (README best practice #4).
  * Visual token budget is configurable: 70/140/280/560/1120 (default 280). For
    MathVision (diagrams + small text) use a HIGH budget. We set it and then VERIFY
    by counting the image placeholder tokens actually inserted into the prompt.

Refs: https://huggingface.co/google/diffusiongemma-26B-A4B-it (README + configs)
      https://ai.google.dev/gemma/docs/diffusiongemma/inference-diffusiongemma-with-hf

Usage (inside the container, see scripts/diffusiongemma_mathvision_demo.slurm):
  python scripts/diffusiongemma_mathvision_demo.py --num 3 --image-tokens 1120
"""

from __future__ import annotations

import argparse
import json
import os

DEFAULT_PARQUET = (
    "/lustre/work/pdl16831/usx56od/Datasets/MathVision/"
    "test-00000-of-00001-3532b8d3f1b4047a.parquet"
)
MODEL_ID = "google/diffusiongemma-26B-A4B-it"
IMAGE_TOKEN_ID = 258880  # config.json: image_token_id (the soft-token placeholder)


def load_samples(parquet, num):
    """Return the first ``num`` MathVision QASamples (image + question + answer).

    Read with pyarrow (not HF ``datasets``) so we don't drag in a huggingface_hub
    version that clashes with the newer one transformers 5.x installs. The
    ``decoded_image`` cell comes back as a struct dict ({bytes, path}) or raw
    bytes, both of which ``mathvision._to_pil`` already handles.
    """
    import pyarrow.parquet as pq

    from src.datasets.dataset_specific import mathvision

    pf = pq.ParquetFile(parquet)
    rows = next(pf.iter_batches(batch_size=num)).to_pylist()
    samples = [mathvision.extract_qa_sample(r) for r in rows[:num]]
    return [s for s in samples if s is not None]


def build_text(sample) -> str:
    """Plain MathVision prompt: question (+ options), ask for a boxed final answer."""
    from src.datasets.dataset_specific import mathvision

    question = mathvision._strip_image_tags(sample.get("question", ""))
    options = sample.get("options") or []
    parts = [question]
    if options:
        lines = [f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)]
        parts.append("Options:\n" + "\n".join(lines))
    parts.append(
        "Solve the problem step by step, then give the final answer as "
        "\\boxed{...} (the option letter if this is multiple choice)."
    )
    return "\n\n".join(parts)


def set_image_budget(processor, n_tokens):
    """Best-effort: request ``n_tokens`` soft tokens per image, defensively.

    The exact attribute path can vary across transformers versions, so we set
    every plausible field and *verify* the effect at tokenization time instead of
    trusting the assignment.
    """
    if n_tokens is None:
        return
    targets = []
    ip = getattr(processor, "image_processor", None)
    for obj, attr in (
        (processor, "image_seq_length"),
        (ip, "image_seq_length"),
        (ip, "max_soft_tokens"),
        (ip, "default_output_length"),
    ):
        if obj is not None and hasattr(obj, attr):
            try:
                setattr(obj, attr, n_tokens)
                targets.append(f"{type(obj).__name__}.{attr}")
            except Exception:  # noqa: BLE001
                pass
    print(f"  requested image budget {n_tokens} via: {targets or 'none found'}")


def main():
    ap = argparse.ArgumentParser(description="DiffusionGemma x MathVision smoke test")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--parquet", type=str, default=DEFAULT_PARQUET)
    ap.add_argument("--model-id", type=str, default=MODEL_ID)
    ap.add_argument("--image-tokens", type=int, default=1120,
                    help="Visual token budget per image (70/140/280/560/1120). "
                         "High budget preserves small text/diagrams in MathVision.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                    help="Enable DiffusionGemma thinking/reasoning mode. NOTE: the "
                         "reasoning trace is long -- pair with a large --max-new-tokens "
                         "(>=1024) or it gets cut off before the boxed answer. Default "
                         "off gives concise, directly-parseable answers.")
    ap.add_argument("--out-dir", type=str, default="diffusiongemma_mathvision_demo_out")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}   Model: {args.model_id}")
    print(f"Gen: image_tokens={args.image_tokens} max_new_tokens={args.max_new_tokens} "
          f"think={args.think}")

    samples = load_samples(args.parquet, args.num)
    print(f"Loaded {len(samples)} MathVision examples")

    print("Loading processor / model (dtype=auto, device_map=auto)...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    print(f"  processor class: {type(processor).__name__}")
    set_image_budget(processor, args.image_tokens)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model_id, dtype="auto", device_map="auto",
    )
    model.eval()

    records = []
    for i, s in enumerate(samples):
        img = s["image"]  # PIL.Image, already RGB (see mathvision._to_pil)
        text = build_text(s)
        img.save(os.path.join(args.out_dir, f"ex{i:02d}_original.png"))

        # Image BEFORE text. Pass the PIL image directly (no squash/crop) -- the
        # Gemma4 image processor handles variable aspect ratio / resolution.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ],
        }]

        tmpl_kwargs = dict(
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        if args.think:
            tmpl_kwargs["enable_thinking"] = True
        try:
            inputs = processor.apply_chat_template(messages, **tmpl_kwargs)
        except TypeError:
            tmpl_kwargs.pop("enable_thinking", None)
            inputs = processor.apply_chat_template(messages, **tmpl_kwargs)
        inputs = inputs.to(model.device)

        # --- VERIFY image handling (don't trust, measure) -------------------
        ids = inputs["input_ids"]
        n_img_tok = int((ids == IMAGE_TOKEN_ID).sum().item())
        pv = inputs.get("pixel_values")
        pv_shape = tuple(pv.shape) if pv is not None else None
        prompt_len = ids.shape[-1]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        # generate() returns a DiffusionGemmaGenerationOutput (a ModelOutput), so
        # out[0] would be the whole 2-D `sequences` tensor -- take .sequences[0].
        seq = out.sequences if hasattr(out, "sequences") else out
        new_tokens = seq[0][prompt_len:]
        full = processor.decode(new_tokens, skip_special_tokens=False)
        clean = processor.decode(new_tokens, skip_special_tokens=True)

        from src.datasets.dataset_specific import mathvision
        parsed = mathvision.parse_answer(clean)
        question = mathvision._strip_image_tags(s.get("question", ""))

        print("\n" + "=" * 72)
        print(f"EXAMPLE {i}  id={s.get('id')}  subject={s.get('subject')} "
              f"level={s.get('level')}")
        print(f"image: original size={img.size}  ->  pixel_values={pv_shape}  "
              f"image_tokens_in_prompt={n_img_tok}")
        print("-" * 72)
        print(f"Q: {question}")
        for j, opt in enumerate(s.get("options") or []):
            print(f"   ({chr(ord('A') + j)}) {opt}")
        print("-" * 72)
        print(f"REFERENCE: {s.get('reference_answer')}")
        print(f"PARSED:    {parsed}")
        print("MODEL (clean):")
        print(clean.strip()[:2000])

        records.append({
            "index": i, "id": s.get("id"), "subject": s.get("subject"),
            "level": s.get("level"), "question": question,
            "options": s.get("options") or [],
            "reference_answer": s.get("reference_answer"),
            "image_size": list(img.size), "pixel_values_shape": pv_shape,
            "image_tokens_in_prompt": n_img_tok, "prompt_len": prompt_len,
            "model_answer_clean": clean, "model_answer_full": full,
            "parsed_answer": parsed,
            "image_original": os.path.join(args.out_dir, f"ex{i:02d}_original.png"),
        })

    out_jsonl = os.path.join(args.out_dir, "results.jsonl")
    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} records to {out_jsonl}")


if __name__ == "__main__":
    main()
