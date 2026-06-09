"""WMT14 German-English dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample:
    """Extract question and answer from a WMT14 example for dataset creation."""
    translation = example.get("translation", example)
    assert isinstance(translation, dict), "Expected 'translation' field to be a dict"
    assert "de" in translation, "WMT example is missing source language 'de'"
    assert "en" in translation, "WMT example is missing target language 'en'"

    return QASample(
        question=str(translation["de"]).strip(),
        reference_answer=str(translation["en"]).strip(),
        source_language="de",
        target_language="en",
    )

# Generation functions
def format_prompt(
    qa_sample: QASample,
    fewshot_prefix: str | None = None,
) -> str:
    """Format a prompt for WMT14 German-English translation."""
    prompt = "Translate the following German text into English, give only the translation.\n\n"
    prompt += fewshot_prefix or ""
    prompt += f"Original: {qa_sample['question']}\nTranslation:"
    return prompt

def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for WMT14 German-English translation."""
    prefix = "Here are a few examples:\n\n" if len(few_shot_examples or []) > 0 else ""
    for example in few_shot_examples or []:
        prefix += f"Original: {example['question']}\nTranslation: {example['reference_answer']}\n\n"
    return prefix


# Labeling functions
def build_judge_prompt(record: dict[str, Any]) -> str:
    """Build a prompt for LLM judging of a WMT14 record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key with model response"
    prompt = "Task : You evaluate the quality of a machine translation from German to English.\n\n"
    prompt += f"German original: {record['question']}\n"
    prompt += f"English translation: {record['raw_answer']}\n"
    prompt += "Ignore stylistic differences. Answer only 0 (critical semantic error or hallucination) or \
        1 (meaning preserved).\nOutput a single digit, nothing else.\n"
    prompt += "Score:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a WMT14 record into a label."""
    return parse_judge_score(text)
