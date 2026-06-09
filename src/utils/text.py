"""Text parsing, normalization, and processing utilities."""

from __future__ import annotations

import re
import string
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    """Strip and collapse whitespace."""
    return _WHITESPACE_RE.sub(" ", str(text).strip())


def normalize_key(text: Any) -> str:
    """Lowercase, collapse whitespace, strip punctuation at edges."""
    return normalize_text(text).lower().strip(string.punctuation + " ")


def parse_text_response(response: Any) -> str:
    """Extract text content from a response that may be a dict or string."""
    if isinstance(response, dict):
        for key in ("text", "content", "response", "answer"):
            if key in response:
                return normalize_text(response[key])
        return normalize_text(str(response))
    return normalize_text(response)


def normalize_aliases(aliases: Any) -> list[str]:
    """Flatten and normalize alias lists."""
    if aliases is None:
        return []
    if isinstance(aliases, str):
        return [normalize_text(aliases)] if aliases.strip() else []
    return [normalize_text(a) for a in aliases if a and str(a).strip()]


def boxed_content(text: str) -> str | None:
    """Extract the final balanced \\boxed{...} content when present."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    chars = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        index += 1
    return None


def parse_math_response(response: Any) -> str:
    """Extract a deterministic final answer from a math-style response."""
    text = str(response).strip()
    marker = re.search(r"####\s*(.*)$", text, flags=re.DOTALL)
    if marker:
        text = marker.group(1).strip()
    else:
        boxed = boxed_content(text)
        if boxed is not None:
            text = boxed
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                text = lines[-1]
    text = re.sub(r"^(?:answer|final answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text.strip(".")


def extract_last_number(text: Any) -> str:
    """Return the final numeric mention after math-answer parsing."""
    normalized = str(parse_math_response(text)).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    return matches[-1] if matches else ""
