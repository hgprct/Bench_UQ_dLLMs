"""Diagnose DiffusionGemma generate() output shape + decoding."""
from __future__ import annotations

import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from src.datasets.dataset_specific import mathvision

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
PARQUET = ("/lustre/work/pdl16831/usx56od/Datasets/MathVision/"
           "test-00000-of-00001-3532b8d3f1b4047a.parquet")


def show(tag, processor, prompt_len, out):
    print(f"\n##### {tag} #####")
    print("type(out):", type(out))
    seq = out.sequences if hasattr(out, "sequences") else out
    print("seq.shape:", tuple(seq.shape), " prompt_len:", prompt_len)
    full = seq[0]
    print("decode FULL seq[0] (skip_special=False), len chars:")
    dec_full = processor.decode(full, skip_special_tokens=False)
    print(repr(dec_full[:1500]))
    print("\ndecode seq[0][prompt_len:] (skip_special=True):")
    print(repr(processor.decode(full[prompt_len:], skip_special_tokens=True)[:1000]))


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto").eval()

    # ---- 1. text-only sanity (README example) ----
    msg = [{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}]
    inp = processor.apply_chat_template(
        msg, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    plen = inp["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=128)
    show("TEXT-ONLY (no think)", processor, plen, out)

    # ---- 2. image + text, thinking on ----
    rows = next(pq.ParquetFile(PARQUET).iter_batches(batch_size=1)).to_pylist()
    s = mathvision.extract_qa_sample(rows[0])
    text = (mathvision._strip_image_tags(s.get("question", "")) +
            "\n\nGive the final answer as \\boxed{...}.")
    for think in (True, False):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": s["image"]},
            {"type": "text", "text": text}]}]
        kw = dict(tokenize=True, add_generation_prompt=True,
                  return_dict=True, return_tensors="pt")
        if think:
            kw["enable_thinking"] = True
        inp = processor.apply_chat_template(messages, **kw).to(model.device)
        plen = inp["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256)
        show(f"IMAGE+TEXT think={think}", processor, plen, out)


if __name__ == "__main__":
    main()
