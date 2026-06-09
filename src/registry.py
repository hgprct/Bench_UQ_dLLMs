"""Strict dataset module registry -- 7 canonical names, no aliases."""

from __future__ import annotations

from types import ModuleType

from src.datasets.dataset_specific import xsum
from src.datasets.dataset_specific import gsm8k, hotpotqa, musique, samsum, triviaqa, wmt14_fr_en, wmt14_de_en

_MODULES: dict[str, ModuleType] = {
    "triviaqa": triviaqa,
    "gsm8k": gsm8k,
    "wmt14_fr_en": wmt14_fr_en,
    "wmt14_de_en": wmt14_de_en,
    "xsum": xsum,
    "samsum": samsum,
    "hotpotqa": hotpotqa,
    "musique": musique,
}


def get_dataset_module(name: str) -> ModuleType:
    """Resolve a canonical dataset name to its adapter module."""
    try:
        return _MODULES[name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Valid: {sorted(_MODULES)}")
