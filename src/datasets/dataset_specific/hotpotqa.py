"""HotpotQA multi-hop in-context QA dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample


# Data building functions
def extract_qa_sample(example: dict[str, Any], id: str) -> QASample | None:
    """Extract question and answer from a HotpotQA example for dataset creation."""
    question = str(example.get("question", "")).strip()
    ground_truth_answer = str(example.get("answer", "")).strip()
    context_raw = example.get("context", None)
    assert context_raw is not None, "Expected 'context' key in example"

    titles_list = context_raw.get("title", [])
    sentences_list = context_raw.get("sentences", [])

    if len(titles_list) == 0:
        print("Expected 'title' list in context to be non-empty")
        return None
    assert len(titles_list) == len(sentences_list), "Expected 'title' and 'sentences' lists to be of the same length"

    context_list = []
    for title, sentences in zip(titles_list, sentences_list):
        context_doc = {"title": title, "sentences": "".join(sentences)}
        context_list.append(context_doc)

    # Join all context documents into a single string
    context_str = "\n\n".join([f"{doc['title']}: {doc['sentences']}" for doc in context_list])

    qid = str(example.get("id", "")).strip() or id
    return QASample(
        id=qid,
        image_id=qid,
        full_prompt=format_prompt(question, context_str),
        question=question,
        ground_truth_answer=ground_truth_answer,
        model_name=None,
        dataset_name="hotpotqa",
        split="validation"
    )


# Generation functions
def format_prompt(question: str, context: str) -> str:
    """Format a prompt for HotpotQA."""
    prompt = "The following are the given documents:\n\n"
    prompt += f"{context}\n\n"
    prompt += "Answer the question based strictly on the provided documents.\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt
