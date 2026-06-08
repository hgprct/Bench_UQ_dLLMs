"""Tests for src.datasets.gsm8k adapter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.gsm8k import (
    extract_final_answer,
    extract_question_and_answer,
    format_prompt,
    normalize_answer,
    parse_response,
)


class TestExtractFinalAnswer:
    def test_with_marker(self):
        assert extract_final_answer("Step 1: ... #### 42") == "42"

    def test_decimal(self):
        assert extract_final_answer("#### 3.14") == "3.14"

    def test_no_marker_fallback(self):
        assert extract_final_answer("The answer is 7") == "7"


class TestParseResponse:
    def test_marker(self):
        assert parse_response("Some work\n#### 100") == "100"

    def test_number_only(self):
        assert parse_response("42") == "42"

    def test_empty(self):
        assert parse_response("no numbers here") == ""


class TestNormalizeAnswer:
    def test_integer(self):
        assert normalize_answer("#### 42") == "42"

    def test_decimal(self):
        assert normalize_answer("#### 3.14") == "3.14"

    def test_trailing_zeros(self):
        assert normalize_answer("#### 3.0") == "3"


class TestExtractQuestionAndAnswer:
    def test_basic(self):
        example = {
            "question": "If you have 3 apples and get 2 more, how many do you have?",
            "answer": "3 + 2 = 5\n#### 5",
        }
        qa = extract_question_and_answer(example)
        assert qa["question"] == "If you have 3 apples and get 2 more, how many do you have?"
        assert qa["reference_answer"] == "5"


class TestFormatPrompt:
    def test_includes_question(self):
        prompt = format_prompt("How many apples?")
        assert "How many apples?" in prompt
        assert "#### [answer]" in prompt

    def test_with_prefix(self):
        prompt = format_prompt("How many?", prefix="Example: ...")
        assert "Example: ..." in prompt
