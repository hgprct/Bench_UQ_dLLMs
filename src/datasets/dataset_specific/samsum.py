"""SamSum dialogue summarization dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract dialogue and summary from a SamSum example for dataset creation."""
    assert "dialogue" in example, "SamSum example is missing 'dialogue' field"
    assert "summary" in example, "SamSum example is missing 'summary' field"
    question = str(example.get("dialogue", "")).strip()
    ground_truth_answer = str(example.get("summary", "")).strip()

    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="samsum",
        split="test"
    )


# Generation functions
def format_prompt(dialogue: str) -> str:
    """Format a prompt for SamSum summarization."""
    instruction = (
        "Provide a summary for the following dialogue. The summary should "
        "(1) be rather short, (2) extract important pieces of information, "
        "(3) include names of interlocutors, (4) be written in the third person."
    )
    return (
        f"{instruction}\n\n"
        f"Here's the dialogue:\n{dialogue}\n"
        f"(End of dialogue).\n\nSummary:"
    )
