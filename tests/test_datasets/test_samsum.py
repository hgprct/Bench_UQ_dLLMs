"""Tests for the samsum adapter (canonical QASample contract)."""

from src.datasets.dataset_specific.samsum import extract_qa_sample, format_prompt


class TestExtractQaSample:
    def test_canonical_fields(self):
        example = {"dialogue": "Alice: Hi!\nBob: Hey!", "summary": "They greet."}
        s = extract_qa_sample(example, "3")
        assert s["id"] == "3"
        assert s["image_id"] == "3"
        assert "Alice: Hi!" in s["question"]
        assert s["ground_truth_answer"] == "They greet."
        assert s["dataset_name"] == "samsum"
        assert "image" not in s

    def test_full_prompt_precomputed(self):
        s = extract_qa_sample({"dialogue": "A: Hi", "summary": "ok"}, "0")
        assert s["full_prompt"] == format_prompt("A: Hi")


class TestFormatPrompt:
    def test_includes_dialogue_and_summary(self):
        prompt = format_prompt("Alice: Hi!\nBob: Hey!")
        assert "dialogue" in prompt.lower()
        assert "Summary:" in prompt
        assert "Alice: Hi!" in prompt
