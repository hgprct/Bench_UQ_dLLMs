"""Diagnostic: what does MMaDA actually 'see' for MathVision images?

MMaDA's understanding path encodes the image with the MAGVIT-v2 VQ tokenizer
(get_code) and feeds those discrete codes to the LLM -- there is no CLIP/SigLIP
feature path. This script reproduces that exact encode and then decodes the same
codes back to pixels (decode_code), so we can visually judge how much fine
detail (numbers, thin lines) survives the 512->32x32 quantization.

Saves, per example: the squashed model-input image and its VQ reconstruction.

Run inside the container:
  python scripts/mmada_vq_reconstruct_check.py --num 10
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MMADA_ROOT = os.path.join(_PROJECT_ROOT, "MMaDA")
for _p in (_MMADA_ROOT, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import MAGVITv2  # noqa: E402
from training.utils import image_transform_squash  # noqa: E402
from src.datasets.dataset_specific import mathvision  # noqa: E402

RESOLUTION = 512


def to_pil(tensor):
    from PIL import Image
    arr = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0) * 255.0
    arr = arr.permute(1, 2, 0).cpu().float().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--parquet", type=str,
                   default="/lustre/work/pdl16831/usx56od/Datasets/MathVision/"
                           "test-00000-of-00001-3532b8d3f1b4047a.parquet")
    p.add_argument("--vq-model", type=str, default="showlab/magvitv2")
    p.add_argument("--out-dir", type=str,
                   default="mmada_mathvision_demo_out/vq_check")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from datasets import Image as HFImage
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=args.parquet, split="train")
    ds = ds.cast_column("decoded_image", HFImage())
    n = min(len(ds), args.num)
    samples = [mathvision.extract_qa_sample(ds[i]) for i in range(n)]
    samples = [s for s in samples if s is not None]

    vq = MAGVITv2.from_pretrained(args.vq_model).to(device).eval()
    vq.requires_grad_(False)

    for i, s in enumerate(samples):
        img = image_transform_squash(s["image"], resolution=RESOLUTION).to(device).unsqueeze(0)
        codes = vq.get_code(img)                       # (1, 1024) in [0, 8191]
        recon = vq.decode_code(codes)[0]               # (3, 512, 512)
        n_unique = int(codes.unique().numel())
        to_pil(img[0]).save(os.path.join(args.out_dir, f"ex{i:02d}_input.png"))
        to_pil(recon).save(os.path.join(args.out_dir, f"ex{i:02d}_recon.png"))
        print(f"EX{i} id={s.get('id'):>3}  codes={tuple(codes.shape)} "
              f"unique_codes={n_unique}/{codes.numel()}  q={s.get('question','')[:50]!r}")

    print(f"\nSaved input/recon pairs to {args.out_dir}/")


if __name__ == "__main__":
    main()
