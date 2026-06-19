"""Strict dataset module registry -- 7 canonical names, no aliases."""

from __future__ import annotations

from types import ModuleType

from src.datasets.dataset_specific import xsum
from src.datasets.dataset_specific import (
    competition_math, drive_bench, gsm8k, hotpotqa, livebench_reasoning,
    mathvision, medmcqa, musique, samsum, triviaqa, vilp, vqa_rad, wmt14_fr_en,
)

_MODULES: dict[str, ModuleType] = {
    "triviaqa": triviaqa,
    "gsm8k": gsm8k,
    "wmt14_fr_en": wmt14_fr_en,
    "xsum": xsum,
    "samsum": samsum,
    "hotpotqa": hotpotqa,
    "musique": musique,
    "mathvision": mathvision,
    "competition_math": competition_math,
    "medmcqa": medmcqa,
    "livebench_reasoning": livebench_reasoning,
    "vqa_rad": vqa_rad,
    "drive_bench": drive_bench,
    "vilp": vilp,
}


def get_dataset_module(name: str) -> ModuleType:
    """Resolve a canonical dataset name to its adapter module."""
    try:
        return _MODULES[name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Valid: {sorted(_MODULES)}")
