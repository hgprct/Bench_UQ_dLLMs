"""MuSiQue multi-hop in-context QA dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample
from src.utils.text import normalize_aliases


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample:
    """Extract question and answer from a MuSiQue example for dataset creation."""
    question = str(example.get("question", "")).strip()
    ground_truth_answer = str(example.get("answer", "")).strip()
    context_raw = example.get("paragraphs", None)
    assert context_raw is not None, "Expected 'context' key in example"

    assert len(context_raw) > 0, "Expected 'paragraphs' to be non-empty"
    assert all("title" in doc and "paragraph_text" in doc for doc in context_raw), \
        "Each document must have 'title' and 'paragraph_text' keys"

    context_list = []
    for doc in context_raw:
        title = doc.get("title", "")
        paragraph_text = doc.get("paragraph_text", "")
        context_doc = {"title": title, "paragraph_text": paragraph_text}
        context_list.append(context_doc)

    context_str = "\n\n".join([f"{doc['title']}: {doc['paragraph_text']}" for doc in context_list])

    return QASample(
        id=id,
        image_id=id,
        full_prompt=format_prompt(question, context_str),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="musique",
        split="train"
    )


# Generation functions
def format_prompt(question: str, context: str) -> str:
    """Format a prompt for MuSiQue."""
    prompt = "The following are the given documents:\n\n"
    prompt += f"{context}\n\n"
    prompt += "Answer the question based strictly on the provided documents.\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt