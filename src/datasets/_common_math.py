"""Math-specific parsing helpers shared by numeric dataset adapters."""

from __future__ import annotations

import re
from typing import Any

from src.datasets._common import normalize_key


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


def normalize_math_answer(answer: Any) -> str:
    """Canonicalize common LaTeX/math surface forms for exact matching."""
    text = parse_math_response(answer)
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\circ", "")
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return normalize_key(text)


def extract_last_number(text: Any) -> str:
    """Return the final numeric mention after math-answer parsing."""
    normalized = str(parse_math_response(text)).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", normalized)
    return matches[-1] if matches else ""
