"""Dataset adapter registry.

Each dataset has a dedicated adapter module under
``src.datasets.dataset_specific`` whose filename matches the :class:`Dataset`
enum value exactly (e.g. ``gsm8k`` -> ``src/datasets/dataset_specific/gsm8k.py``).
This module is the single place that turns a dataset key into its adapter.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from src.config import Dataset


def get_dataset_module(dataset_key: str) -> ModuleType:
    """Return the adapter module for *dataset_key*.

    Validates the key against :class:`Dataset` first so an unknown dataset fails
    with a clear error rather than an opaque ``ModuleNotFoundError``.
    """
    try:
        Dataset(dataset_key)
    except ValueError as exc:
        valid = ", ".join(d.value for d in Dataset)
        raise ValueError(
            f"Unknown dataset {dataset_key!r}. Valid datasets: {valid}"
        ) from exc
    return importlib.import_module(f"src.datasets.dataset_specific.{dataset_key}")
