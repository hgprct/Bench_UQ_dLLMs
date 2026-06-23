"""Tests for the triviaqa adapter (canonical QASample contract)."""

from src.datasets.dataset_specific.triviaqa import extract_qa_sample, format_prompt


class TestExtractQaSample:
    def test_aliases_parsed_into_ground_truth(self):
        example = {"id": "wh_1", "question": "Capital of France?",
                   "answer": "['Paris', 'paris']"}
        s = extract_qa_sample(example, "0")
        assert s["ground_truth_answer"] == ["Paris", "paris"]
        assert s["question"] == "Capital of France?"
        assert s["dataset_name"] == "triviaqa"

    def test_native_id_overrides_passed_id(self):
        example = {"id": "wh_4198", "question": "Q?", "answer": "['A']"}
        s = extract_qa_sample(example, "99")
        assert s["id"] == "wh_4198"
        assert s["image_id"] == "wh_4198"

    def test_missing_answer_returns_none(self):
        assert extract_qa_sample({"id": "x", "question": "Q?", "answer": ""}, "0") is None

    def test_missing_question_returns_none(self):
        assert extract_qa_sample({"id": "x", "question": "", "answer": "['A']"}, "0") is None


class TestFormatPrompt:
    def test_basic(self):
        prompt = format_prompt("What is 2+2?")
        assert "Question: What is 2+2?" in prompt
        assert "Answer:" in prompt
