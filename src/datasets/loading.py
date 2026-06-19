"""Dataset loading: HuggingFace and local file support."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_hf_dataset(
    hf_name: str,
    split: str,
    config_name: str | None = None,
    token: str | None = None,
) -> Any:
    """Load a dataset from HuggingFace Hub."""
    from datasets import load_dataset
    if config_name:
        return load_dataset(hf_name, config_name, split=split, token=token)
    return load_dataset(hf_name, split=split, token=token)


def load_local_dataset(
    path: str | Path,
    file_format: str | None = None,
) -> Any:
    """Load a dataset from a local file (jsonl, json, or parquet).

    The format is inferred from the file extension unless *file_format* is given.
    Parquet returns a ``datasets.Dataset`` (supports ``len`` and integer indexing
    like an HF dataset, and decodes Image/struct features); json/jsonl return a
    ``list[dict]``. Both are consumed identically by ``build_raw_prompts``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    fmt = (file_format or path.suffix.lstrip(".")).lower()
    if fmt == "jsonl":
        from src.utils.io import read_jsonl
        return read_jsonl(path)
    elif fmt == "json":
        from src.utils.io import read_json
        data = read_json(path)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON array in {path}")
    elif fmt in ("parquet", "pq"):
        from datasets import Dataset
        return Dataset.from_parquet(str(path))
    else:
        raise ValueError(f"Unsupported file format: {fmt!r} for {path}")
