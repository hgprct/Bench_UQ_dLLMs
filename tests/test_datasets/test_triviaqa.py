"""Tests for src.datasets.triviaqa adapter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.triviaqa import (
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
            "question": "What is the capital of France?",
            "answer": {"value": "Paris", "aliases": ["paris", "City of Light"]},
        }
        qa = extract_question_and_answer(example)
        assert qa["question"] == "What is the capital of France?"
        assert qa["reference_answer"] == "Paris"
        assert "paris" in qa["aliases"]

    def test_missing_answer(self):
        example = {"question": "Test?", "answer": {}}
        qa = extract_question_and_answer(example)
        assert qa["reference_answer"] == ""


class TestFormatPrompt:
    def test_basic(self):
        prompt = format_prompt("What is 2+2?")
        assert "Question: What is 2+2?" in prompt
        assert "Answer:" in prompt

    def test_with_prefix(self):
        prompt = format_prompt("What is 2+2?", prefix="Some context")
        assert "Some context" in prompt
        assert "Question: What is 2+2?" in prompt


class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("The Eiffel Tower") == "eiffel tower"

    def test_articles_removed(self):
        assert normalize_answer("a big dog") == "big dog"

    def test_punctuation_stripped(self):
        assert normalize_answer("hello!") == "hello"

    def test_underscores(self):
        assert normalize_answer("new_york") == "new york"


class TestExtractShortAnswer:
    def test_plain_answer(self):
        assert extract_short_answer("Paris") == "Paris"

    def test_answer_prefix(self):
        assert extract_short_answer("The answer is Paris") == "Paris"

    def test_final_answer_prefix(self):
        assert extract_short_answer("Final answer: Paris") == "Paris"


class TestBuildJudgePrompt:
    def test_contains_question_and_candidate(self):
        record = {
            "question": "Capital of France?",
            "reference_answer": "Paris",
            "aliases": ["Paris", "City of Light"],
            "final": {"answer": "Paris"},
        }
        prompt = build_judge_prompt(record)
        assert "Capital of France?" in prompt
        assert "Paris" in prompt

    def test_uses_aliases_as_references(self):
        record = {
            "question": "Q?",
            "reference_answer": "Paris",
            "aliases": ["Paris", "City of Light"],
            "final": {"answer": "Paris"},
        }
        prompt = build_judge_prompt(record)
        assert "City of Light" in prompt

    def test_output_format(self):
        record = {"question": "Q?", "reference_answer": "A", "final": {"answer": "A"}}
        prompt = build_judge_prompt(record)
        assert "Score:" in prompt


class TestParseJudgeOutput:
    def test_single_digit_one(self):
        assert parse_judge_output("1") is True

    def test_single_digit_zero(self):
        assert parse_judge_output("0") is False

    def test_score_one(self):
        assert parse_judge_output("Reasoning: The answer matches.\nScore: 1") is True

    def test_score_zero(self):
        assert parse_judge_output("Reasoning: The answer does not match.\nScore: 0") is False

    def test_unparseable_returns_none(self):
        assert parse_judge_output("I think maybe") is None

    def test_no_score_returns_none(self):
        assert parse_judge_output("Reasoning: ok") is None

    def test_freeform_with_score(self):
        assert parse_judge_output("The model said Paris which is correct.\nScore: 1") is True

    def test_leading_whitespace(self):
        assert parse_judge_output(" 1") is True
        assert parse_judge_output(" 0") is False

    def test_leading_newline(self):
        assert parse_judge_output("\n1") is True
        assert parse_judge_output("\n0") is False

    def test_trailing_eos_text(self):
        assert parse_judge_output("1\n") is True
        assert parse_judge_output("0\n\n") is False

    def test_digit_with_period(self):
        assert parse_judge_output("1.") is True
        assert parse_judge_output("0.") is False

    def test_multi_digit_number_returns_none(self):
        assert parse_judge_output("10") is None
        assert parse_judge_output("100") is None

    def test_none_input(self):
        assert parse_judge_output(None) is None

    def test_empty_string(self):
        assert parse_judge_output("") is None
        assert parse_judge_output("   ") is None
