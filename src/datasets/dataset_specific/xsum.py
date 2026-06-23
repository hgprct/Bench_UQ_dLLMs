"""XSum extreme summarization dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract document and summary from an XSum example for dataset creation."""
    assert "document" in example, "XSum example is missing 'document' field"
    assert "summary" in example, "XSum example is missing 'summary' field"
    question = str(example.get("document", "")).strip()
    ground_truth_answer = str(example.get("summary", "")).strip()

    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="xsum",
        split="test"
    )


# Generation functions
def format_prompt(document: str) -> str:
    """Format a prompt for XSum summarization."""
    instruction = "Provide a short one-sentence summary for the following text."
    return (
        f"{instruction}\n\n"
        f"Here's the text:\n{document}\n"
        f"(End of text).\n\nSummary:"
    )