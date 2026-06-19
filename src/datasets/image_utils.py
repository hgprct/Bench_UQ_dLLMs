"""Shared helpers for coercing HuggingFace image cells into PIL RGB images."""

from __future__ import annotations

import io
from typing import Any


def to_pil(value: Any):
    """Coerce a HF image cell (PIL, {"bytes": ...}, path, or bytes) to RGB PIL.

    Returns ``None`` when *value* is ``None`` so callers can skip imageless rows.
    """
    from PIL import Image

    if value is None:
        return None
    if isinstance(value, Image.Image):
        img = value
    elif isinstance(value, dict) and value.get("bytes") is not None:
        img = Image.open(io.BytesIO(value["bytes"]))
    elif isinstance(value, (str, bytes)):
        img = Image.open(value if isinstance(value, str) else io.BytesIO(value))
    else:
        raise TypeError(f"Unsupported image cell type: {type(value)!r}")
    return img.convert("RGB")
