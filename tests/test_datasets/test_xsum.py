"""Tests for the xsum adapter (canonical QASample contract)."""

from src.datasets.dataset_specific.xsum import extract_qa_sample, format_prompt


class TestExtractQaSample:
    def test_canonical_fields(self):
        example = {"document": "The cat sat on the mat.", "summary": "A cat sat."}
        s = extract_qa_sample(example, "12")
        assert s["id"] == "12"
        assert s["image_id"] == "12"
        assert s["question"] == "The cat sat on the mat."
        assert s["ground_truth_answer"] == "A cat sat."
        assert s["dataset_name"] == "xsum"
        assert s["split"] == "test"
        assert "image" not in s

    def test_full_prompt_precomputed(self):
        s = extract_qa_sample({"document": "Some text.", "summary": "sum"}, "0")
        assert s["full_prompt"] == format_prompt("Some text.")


class TestFormatPrompt:
    def test_includes_text_and_summary(self):
        prompt = format_prompt("The cat sat on the mat.")
        assert "summary" in prompt.lower()
        assert "Summary:" in prompt
        assert "The cat sat on the mat." in prompt
