"""Tests for src.datasets.xsum adapter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.xsum import (
    build_judge_prompt,
    extract_question_and_answer,
    extract_short_answer,
    format_prompt,
    normalize_answer,
    parse_judge_output,
)


class TestExtractQuestionAndAnswer:
    def test_basic(self):
        example = {
            "document": "The cat sat on the mat. It was a sunny day.",
            "summary": "A cat sat on a mat on a sunny day.",
            "id": "123",
        }
        qa = extract_question_and_answer(example)
        assert qa["question"] == "The cat sat on the mat. It was a sunny day."
        assert qa["reference_answer"] == "A cat sat on a mat on a sunny day."
        assert qa["metadata"]["task_type"] == "xsum"

    def test_missing_summary(self):
        example = {"document": "Some text.", "summary": ""}
        qa = extract_question_and_answer(example)
        assert qa["reference_answer"] == ""

    def test_missing_document_returns_none(self):
        example = {"document": "", "summary": "A summary."}
        assert extract_question_and_answer(example) is None


class TestFormatPrompt:
    def test_basic(self):
        prompt = format_prompt("The cat sat on the mat.")
        assert "text" in prompt.lower()
        assert "Summary:" in prompt
        assert "The cat sat on the mat." in prompt

    def test_includes_instruction(self):
        prompt = format_prompt("Some article text.")
        assert "summary" in prompt.lower()

    def test_with_prefix(self):
        prompt = format_prompt("Article text.", prefix="Context here")
        assert "Context here" in prompt


class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("A cat sat on a mat.") == "a cat sat on a mat."

    def test_whitespace(self):
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"


class TestExtractShortAnswer:
    def test_plain(self):
        assert extract_short_answer("A cat sat.") == "A cat sat."

    def test_summary_prefix(self):
        assert extract_short_answer("Summary: A cat sat.") == "A cat sat."


class TestBuildJudgePrompt:
    def test_contains_source_and_candidate(self):
        record = {
            "question": "The cat sat on the mat.",
            "reference_answer": "A cat sat.",
            "final": {"answer": "Cat on mat."},
        }
        prompt = build_judge_prompt(record)
        assert "The cat sat on the mat." in prompt
        assert "Cat on mat." in prompt

    def test_output_format(self):
        record = {"question": "text", "final": {"answer": "summary"}}
        prompt = build_judge_prompt(record)
        assert "Score:" in prompt


class TestParseJudgeOutput:
    def test_single_digit_one(self):
        assert parse_judge_output("1") is True

    def test_single_digit_zero(self):
        assert parse_judge_output("0") is False

    def test_score_one(self):
        assert parse_judge_output("Reasoning: Fully consistent.\nScore: 1") is True

    def test_score_zero(self):
        assert parse_judge_output("Reasoning: Contains hallucination.\nScore: 0") is False

    def test_unparseable_returns_none(self):
        assert parse_judge_output("I'm not sure about this") is None

    def test_none_input(self):
        assert parse_judge_output(None) is None

    def test_no_score_returns_none(self):
        assert parse_judge_output("Reasoning: ok") is None
