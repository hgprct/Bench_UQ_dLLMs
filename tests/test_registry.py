"""Tests for src.registry -- strict 7-entry dataset registry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import get_dataset_module


class TestGetDatasetModule:
    def test_triviaqa(self):
        mod = get_dataset_module("triviaqa")
        assert hasattr(mod, "extract_question_and_answer")
        assert hasattr(mod, "format_prompt")

    def test_gsm8k(self):
        mod = get_dataset_module("gsm8k")
        assert hasattr(mod, "extract_question_and_answer")

    def test_wmt14(self):
        mod = get_dataset_module("wmt14_fr_en")
        assert hasattr(mod, "extract_question_and_answer")

    def test_unknown_raises(self):
        try:
            get_dataset_module("unknown_dataset")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unknown_dataset" in str(e)
            assert "triviaqa" in str(e)

    def test_xsum(self):
        mod = get_dataset_module("xsum")
        assert hasattr(mod, "extract_question_and_answer")
        assert hasattr(mod, "format_prompt")

    def test_samsum(self):
        mod = get_dataset_module("samsum")
        assert hasattr(mod, "extract_question_and_answer")
        assert hasattr(mod, "format_prompt")

    def test_old_alias_not_supported(self):
        """Old aliases like 'mandarjoshi/trivia_qa' must not work."""
        try:
            get_dataset_module("mandarjoshi/trivia_qa")
            assert False, "Old alias should not be supported"
        except ValueError:
            pass

    def test_hotpotqa(self):
        mod = get_dataset_module("hotpotqa")
        assert hasattr(mod, "extract_question_and_answer")
        assert hasattr(mod, "format_prompt")
        assert hasattr(mod, "build_judge_prompt")
        assert hasattr(mod, "parse_judge_output")

    def test_musique(self):
        mod = get_dataset_module("musique")
        assert hasattr(mod, "extract_question_and_answer")
        assert hasattr(mod, "format_prompt")
        assert hasattr(mod, "build_judge_prompt")
        assert hasattr(mod, "parse_judge_output")

    def test_all_entries(self):
        valid = ["triviaqa", "gsm8k", "wmt14_fr_en", "xsum", "samsum", "hotpotqa", "musique"]
        for name in valid:
            get_dataset_module(name)
