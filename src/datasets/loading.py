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
    file_format: str = "jsonl",
) -> list[dict[str, Any]]:
    """Load a dataset from a local file (jsonl or json)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if file_format == "jsonl":
        from src.utils.io import read_jsonl
        return read_jsonl(path)
    elif file_format == "json":
        from src.utils.io import read_json
        data = read_json(path)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON array in {path}")
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
