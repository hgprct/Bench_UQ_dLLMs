"""HotpotQA multi-hop in-context QA dataset adapter."""

from __future__ import annotations

from typing import Any

from src.datasets.qa_pair import QASample
from src.labeling.judge import parse_judge_score


# Data building functions
def extract_qa_sample(example: dict[str, Any]) -> QASample:
    """Extract question and answer from a HotpotQA example for dataset creation."""
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
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

    return QASample(
        question=question,
        reference_answer=answer,
        source_documents=context_list,
        id=str(example.get("id", None))
    )


# Generation functions
def format_prompt(
    qa_sample: QASample,
    fewshot_prefix: str | None = None,
) -> str:
    """Format a prompt for HotpotQA."""
    prompt = "The following are the given documents:\n\n"

    for i, doc in enumerate(qa_sample.get("source_documents", [])):
        title = doc.get("title", "")
        sentences = doc.get("sentences", [])
        prompt += f"Document {i}: {title}\n"
        prompt +=  sentences + "\n\n"

    prompt += "Answer the question based strictly on the provided documents.\n\n"

    if fewshot_prefix:
        prompt += fewshot_prefix

    prompt += f"Question: {qa_sample['question']}\nAnswer:"
    return prompt


def few_shot_prefix(few_shot_examples: list[QASample] | None = None) -> str:
    """Format a few-shot prefix for HotpotQA."""
    prefix = ""
    for example in few_shot_examples or []:
        prefix += f"Question: {example['question']}\nAnswer: {example['reference_answer']}\n\n"
    return prefix


# Labeling functions
def build_judge_prompt(record: QASample) -> str:
    """Build a prompt for LLM judging of a HotpotQA record."""
    assert "question" in record, "Record must contain 'question' key"
    assert "reference_answer" in record, "Record must contain 'reference_answer' key"
    assert "raw_answer" in record, "Record must contain 'raw_answer' key with model response"
    
    prompt = "Task : You evaluate whether the model response correctly answers the question.\n"
    prompt += "Accept conversational hedges if the core answer matches any reference alias.\n"
    prompt += f"Question: {record['question']}\n"
    prompt += f"Acceptable response: {record['reference_answer']}\n"
    prompt += f"Model response: {record['raw_answer']}\n"
    prompt += "Answer only 0 (incorrect or uncommitted) or 1 (correct).\n"
    prompt += "Output a single digit, nothing else.\n"
    prompt += "Score:"
    return prompt


def parse_judge_output(text: Any) -> bool | None:
    """Parse the output of an LLM judge for a HotpotQA record into a label."""
    return parse_judge_score(text)
