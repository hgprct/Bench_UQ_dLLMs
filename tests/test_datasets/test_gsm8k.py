"""Tests for the gsm8k adapter (canonical QASample contract)."""

from src.datasets.dataset_specific.gsm8k import extract_qa_sample, format_prompt


class TestExtractQaSample:
    def test_canonical_fields(self):
        example = {"question": "How many apples?", "answer": "5 apples\n#### 5"}
        s = extract_qa_sample(example, "7")
        assert s["id"] == "7"
        assert s["image_id"] == "7"          # text-only: image_id mirrors id
        assert s["question"] == "How many apples?"
        assert s["ground_truth_answer"] == "5 apples\n#### 5"
        assert s["dataset_name"] == "gsm8k"
        assert s["split"] == "test"
        assert s["model_name"] is None
        assert "image" not in s              # never carries an image

    def test_full_prompt_precomputed(self):
        s = extract_qa_sample({"question": "Q?", "answer": "#### 1"}, "0")
        assert s["full_prompt"] == format_prompt("Q?")
        assert "Q?" in s["full_prompt"]


class TestFormatPrompt:
    def test_includes_question_and_answer_marker(self):
        prompt = format_prompt("How many apples?")
        assert "How many apples?" in prompt
        assert "#### [answer]" in prompt
