"""Tests for the output/disk layer (images + clean results JSONL)."""

import json

from PIL import Image

from src.datasets import outputs


def _sample(**over):
    base = {
        "id": "0", "image_id": "0", "full_prompt": "FP", "question": "Q?",
        "ground_truth_answer": "A", "greedy_answer": "G", "model_name": "LLaDA",
        "dataset_name": "gsm8k", "split": "test",
    }
    base.update(over)
    return base


class TestResultsJsonl:
    def test_exactly_canonical_fields(self, tmp_path):
        path = outputs.write_results_jsonl("gsm8k", "LLaDA", [_sample()], root=tmp_path)
        assert path == tmp_path / "gsm8k" / "LLaDA.jsonl"
        rec = json.loads(path.read_text().splitlines()[0])
        assert sorted(rec) == sorted(outputs.RESULT_FIELDS)
        assert rec["greedy_answer"] == "G" and rec["model_name"] == "LLaDA"

    def test_missing_fields_default_to_none(self, tmp_path):
        path = outputs.write_results_jsonl(
            "gsm8k", "LLaDA", [{"id": "0", "question": "Q?"}], root=tmp_path)
        rec = json.loads(path.read_text().splitlines()[0])
        assert rec["greedy_answer"] is None and rec["ground_truth_answer"] is None

    def test_image_field_not_serialised(self, tmp_path):
        s = _sample(image=Image.new("RGB", (4, 4)))
        path = outputs.write_results_jsonl("gsm8k", "LLaDA", [s], root=tmp_path)
        rec = json.loads(path.read_text().splitlines()[0])
        assert "image" not in rec


class TestImages:
    def test_save_dedup_and_load_roundtrip(self, tmp_path):
        s = _sample(image_id="42", image=Image.new("RGB", (8, 8), (10, 20, 30)))
        assert outputs.save_images("mathvision", [s], root=tmp_path) == 1
        assert outputs.image_path("mathvision", "42", tmp_path).is_file()
        # re-run reuses the file rather than rewriting it
        assert outputs.save_images("mathvision", [s], root=tmp_path) == 0
        loaded = outputs.load_image("mathvision", "42", tmp_path)
        assert loaded is not None and loaded.size == (8, 8)

    def test_text_only_samples_save_nothing(self, tmp_path):
        assert outputs.save_images("gsm8k", [_sample()], root=tmp_path) == 0

    def test_load_missing_image_returns_none(self, tmp_path):
        assert outputs.load_image("gsm8k", "nope", tmp_path) is None
