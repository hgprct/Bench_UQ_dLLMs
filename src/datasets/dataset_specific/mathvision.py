"""MathVision dataset adapter (multimodal math reasoning).

HF dataset: MathLLMs/MathVision (split "test", 3040 rows).

Row schema:
  id            : str
  question      : str            (may contain "<image1>" placeholders)
  options       : list[str]      (empty for free-form, else multiple choice)
  image         : str            relative path, e.g. "images/1.jpg"
  decoded_image : PIL.Image      the actual image (HF Image feature)
  answer        : str            option letter (MC) or final value (free-form)
  solution      : str            worked solution
  level         : int
  subject       : str

The image is carried inside the QASample (``image`` field) and consumed by the
MMaDA mmu generation path; ``format_prompt`` only builds the text prompt. The
"<image1>" placeholder is stripped because MMaDA supplies the image through
dedicated <|soi|>/<|eoi|> tokens rather than an inline marker.
"""

from __future__ import annotations

import io
import re
from typing import Any

from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score

_IMAGE_TAG_RE = re.compile(r"<image\d+>")
_BOXED_RE = re.compile(r"\\boxed\{")

# MMaDA-8B-MixCoT emits chain-of-thought (a <think>...</think> block) when the
# user turn opens with an explicit reasoning instruction. This preamble is
# prepended to the question, matching the reference demo in MMaDA/app.py and the
# validated native scripts.
MIXCOT_REASONING_PREAMBLE = (
    "Answer the following question about the image, "
    "explaining your reasoning step by step."
)


def _to_pil(value: Any):
    """Coerce a HF decoded_image cell into a PIL.Image (RGB)."""
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
        raise TypeError(f"Unsupported decoded_image type: {type(value)!r}")
    return img.convert("RGB")


def _strip_image_tags(text: str) -> str:
    out = _IMAGE_TAG_RE.sub("", str(text))
    out = re.sub(r"[ \t]{2,}", " ", out)        # collapse runs of spaces
    out = re.sub(r" +([?.,;:!])", r"\1", out)    # drop space left before punctuation
    return out.strip()


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample | None:
    """Extract a multimodal QA sample from a MathVision row."""
    assert "question" in example, "MathVision example is missing 'question' field"
    image = _to_pil(example.get("decoded_image"))
    if image is None:
        return None

    options = list(example.get("options") or [])
    answer = str(example.get("answer", "")).strip()

    return QASample(
        question=str(example.get("question", "")).strip(),
        reference_answer=answer,
        raw_reference_answer=str(example.get("solution", "")).strip(),
        options=options,
        image=image,
        subject=str(example.get("subject", "")).strip(),
        level=str(example.get("level", "")),
        id=str(example.get("id", "")),
        task_type="multimodal_math",
    )


# Generation functions
def format_prompt(
    qa_sample: QASample,
    fewshot_prefix: str | None = None,
) -> str:
    """Build the text prompt for a MathVision question (image added separately)."""
    question = _strip_image_tags(qa_sample.get("question", ""))
    options = qa_sample.get("options") or []

    parts: list[str] = [question]
    if options:
        lines = [f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)]
        parts.append("Options:\n" + "\n".join(lines))

    body = "\n\n".join(parts)
    return f"{MIXCOT_REASONING_PREAMBLE}\n\n{(fewshot_prefix or '')}{body}"


def parse_answer(raw_answer: str) -> str:
    """Extract the final answer from a MixCoT response.

    Preference order: (1) the content of the last ``\\boxed{...}`` if present;
    (2) otherwise whatever follows the last ``</think>`` tag -- MixCoT puts its
    answer there and rarely emits ``\\boxed{}`` on MathVision; (3) otherwise the
    stripped text. The judge handles MC-letter vs free-form normalisation.
    """
    text = str(raw_answer)
    matches = list(_BOXED_RE.finditer(text))
    if matches:
        start = matches[-1].end()
        depth = 1
        out: list[str] = []
        for ch in text[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(ch)
        return "".join(out).strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text.strip()


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of a MathVision record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key"
    options = record.get("options") or []
    final = parse_answer(record["raw_answer"])
    prompt = "Task : You evaluate whether the model solved the math problem correctly.\n"
    prompt += f"Question: {_strip_image_tags(record['question'])}\n"
    if options:
        opts = "\n".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(options))
        prompt += f"Options:\n{opts}\n"
    prompt += f"Correct answer: {record.get('reference_answer', '')}\n"
    prompt += f"Model's final answer: {final}\n"
    prompt += "Answer 1 if the model's final answer is correct (matches the option "
    prompt += "letter or the value), else 0.\n"
    prompt += "Output a single digit, nothing else.\nScore:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a MathVision record into a label."""
    return parse_judge_score(text)
