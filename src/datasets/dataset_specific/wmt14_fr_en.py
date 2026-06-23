"""WMT14 French-English dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample:
    """Extract question and answer from a WMT14 example for dataset creation."""
    translation = example.get("translation", example)
    assert isinstance(translation, dict), "Expected 'translation' field to be a dict"
    assert "fr" in translation, "WMT example is missing source language 'fr'"
    assert "en" in translation, "WMT example is missing target language 'en'"
    question = str(translation.get("fr", "")).strip()
    ground_truth_answer = str(translation.get("en", "")).strip()

    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="wmt14_fr_en",
        split="test"
    )



# Generation functions
def format_prompt(original: str) -> str:
    """Format a prompt for WMT14 French-English translation."""
    prompt = "Translate the following French text into English, give only the translation.\n\n"
    prompt += f"Original: {original}\nTranslation:"
    return prompt