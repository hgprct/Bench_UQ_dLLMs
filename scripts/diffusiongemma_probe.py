"""Environment probe for running google/diffusiongemma-26B-A4B-it.

Prints everything needed to decide how to run DiffusionGemma on this cluster:
arch / GPU, transformers+torch versions, whether the diffusion_gemma model
classes are importable, outbound internet from the compute node, and whether
the weights are already in the HF cache. Run via scripts/diffusiongemma_probe.slurm.
"""

from __future__ import annotations

import os
import platform
import sys

MODEL_ID = "google/diffusiongemma-26B-A4B-it"


def _hr(title):
    print("\n" + "-" * 60 + f"\n{title}\n" + "-" * 60)


def main():
    _hr("platform / python")
    print("machine     :", platform.machine())
    print("python      :", sys.version.split()[0])
    print("PYTHONPATH  :", os.environ.get("PYTHONPATH", ""))

    _hr("torch / GPU")
    try:
        import torch

        print("torch       :", torch.__version__)
        print("cuda avail  :", torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  gpu{i}: {p.name}  {p.total_memory/1e9:.0f} GB")
    except Exception as e:  # noqa: BLE001
        print("torch import FAILED:", repr(e))

    _hr("transformers + diffusion_gemma support")
    try:
        import transformers

        print("transformers:", transformers.__version__)
        from transformers import AutoConfig  # noqa: F401
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        print("diffusion_gemma in CONFIG_MAPPING:",
              "diffusion_gemma" in CONFIG_MAPPING_NAMES)
        for name in ("DiffusionGemmaForBlockDiffusion", "Gemma4Processor",
                     "Gemma4ImageProcessor", "AutoProcessor"):
            try:
                mod = __import__("transformers", fromlist=[name])
                getattr(mod, name)
                print(f"  import {name}: OK")
            except Exception as e:  # noqa: BLE001
                print(f"  import {name}: FAIL ({e.__class__.__name__})")
    except Exception as e:  # noqa: BLE001
        print("transformers import FAILED:", repr(e))

    _hr("outbound internet from compute node")
    import urllib.request

    for url in ("https://huggingface.co", "https://pypi.org"):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as r:
                print(f"  {url} -> {r.status}")
        except Exception as e:  # noqa: BLE001
            print(f"  {url} -> NO ({e.__class__.__name__}: {e})")

    _hr("HF cache")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print("HF_HOME     :", hf_home)
    print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
    cache_dir = os.path.join(hf_home, "hub",
                             "models--google--diffusiongemma-26B-A4B-it")
    print("model cached:", os.path.isdir(cache_dir), "->", cache_dir)


if __name__ == "__main__":
    main()
