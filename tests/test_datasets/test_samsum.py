"""Tests for src.datasets.samsum adapter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from datasets.samsum import (
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
            "dialogue": "Alice: Hi!\nBob: Hey, how are you?\nAlice: Good, thanks!",
            "summary": "Alice and Bob greet each other.",
            "id": "42",
        }
        qa = extract_question_and_answer(example)
        assert "Alice: Hi!" in qa["question"]
        assert qa["reference_answer"] == "Alice and Bob greet each other."
        assert qa["metadata"]["task_type"] == "samsum"

    def test_missing_summary(self):
        example = {"dialogue": "A: Hello\nB: Hi", "summary": ""}
        qa = extract_question_and_answer(example)
        assert qa["reference_answer"] == ""

    def test_missing_dialogue_returns_none(self):
        example = {"dialogue": "", "summary": "A summary."}
        assert extract_question_and_answer(example) is None


class TestFormatPrompt:
    def test_basic(self):
        prompt = format_prompt("Alice: Hi!\nBob: Hey!")
        assert "dialogue" in prompt.lower()
        assert "Summary:" in prompt
        assert "Alice: Hi!" in prompt

    def test_includes_instruction(self):
        prompt = format_prompt("Some dialogue.")
        assert "summary" in prompt.lower()

    def test_with_prefix(self):
        prompt = format_prompt("A: Hi\nB: Bye", prefix="Context")
        assert "Context" in prompt


class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("They greet each other.") == "they greet each other."

    def test_whitespace(self):
        assert normalize_answer("  extra   spaces  ") == "extra spaces"


class TestExtractShortAnswer:
    def test_plain(self):
        assert extract_short_answer("They talk.") == "They talk."

    def test_summary_prefix(self):
        assert extract_short_answer("Summary: They talk.") == "They talk."


class TestBuildJudgePrompt:
    def test_contains_source_and_candidate(self):
        record = {
            "question": "Alice: Hi!\nBob: Hey!",
            "reference_answer": "Alice greets Bob.",
            "final": {"answer": "They say hello."},
        }
        prompt = build_judge_prompt(record)
        assert "Alice: Hi!" in prompt
        assert "They say hello." in prompt

    def test_dialogue_keyword(self):
        record = {"question": "A: Hi", "final": {"answer": "summary"}}
        prompt = build_judge_prompt(record)
        assert "dialogue" in prompt.lower()


class TestParseJudgeOutput:
    def test_single_digit_one(self):
        assert parse_judge_output("1") is True

    def test_single_digit_zero(self):
        assert parse_judge_output("0") is False

    def test_score_one(self):
        assert parse_judge_output("Reasoning: Consistent summary.\nScore: 1") is True

    def test_score_zero(self):
        assert parse_judge_output("Reasoning: Contradicts dialogue.\nScore: 0") is False

    def test_unparseable_returns_none(self):
        assert parse_judge_output("maybe") is None

    def test_none_input(self):
        assert parse_judge_output(None) is None

    def test_no_score_returns_none(self):
        assert parse_judge_output("Reasoning: ok") is None
