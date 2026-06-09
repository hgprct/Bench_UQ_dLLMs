"""Tests for the labeling module."""

from __future__ import annotations

from labeling import label_records, unlabeled_prompt_ids


class FakeEvaluator:
    def __init__(self, answers):
        self._answers = list(answers)
        self._call_count = 0

    def predict_batch(self, prompts, temperature=0.0, max_tokens=1):
        batch = self._answers[self._call_count:self._call_count + len(prompts)]
        self._call_count += len(prompts)
        return batch


class FakeExactMatchModule:
    """Simulates a dataset adapter that uses exact_match labeling (like GSM8K)."""

    DEFAULT_LABEL_METHOD = "exact_match"
    LABEL_GPU_MODE = "cpu"

    @staticmethod
    def normalize_answer(answer):
        return str(answer).strip().lower()

    @staticmethod
    def parse_response(answer):
        return str(answer).strip()

    @staticmethod
    def exact_match_correct(answer, references):
        norm = str(answer).strip().lower()
        return any(norm == str(r).strip().lower() for r in references)

    @staticmethod
    def label_record(record, evaluator=None):
        final = record.get("final", {})
        answer = str(final.get("answer", final.get("response", "")) or "")
        parsed = FakeExactMatchModule.normalize_answer(answer)
        ref = record.get("reference_answer", "")
        all_refs = record.get("aliases", [ref]) if record.get("aliases") else [ref]
        return FakeExactMatchModule.exact_match_correct(parsed, all_refs)

    @staticmethod
    def label_records_batch(records, evaluator=None):
        return [FakeExactMatchModule.label_record(r) for r in records], set()


class FakeJudgeModule:
    """Simulates a dataset adapter that uses llm_judge labeling (like TriviaQA)."""

    DEFAULT_LABEL_METHOD = "llm_judge"
    LABEL_GPU_MODE = "gpu"

    @staticmethod
    def build_judge_prompt(record):
        return f"Is '{record.get('final', {}).get('answer', '')}' correct? yes or no"

    @staticmethod
    def parse_judge_output(text):
        text = str(text).strip().lower()
        if "yes" in text:
            return True
        if "no" in text:
            return False
        return None

    @staticmethod
    def _response_text(record):
        final = record.get("final", {})
        return str(final.get("answer", final.get("response", "")) or "")

    @staticmethod
    def label_record(record, evaluator=None):
        if not FakeJudgeModule._response_text(record).strip():
            return False
        prompt = FakeJudgeModule.build_judge_prompt(record)
        outputs = evaluator.predict_batch([prompt], temperature=0.0, max_tokens=16)
        return FakeJudgeModule.parse_judge_output(outputs[0])

    @staticmethod
    def label_records_batch(records, evaluator=None):
        labels = [None] * len(records)
        judge_indices = []
        for i, record in enumerate(records):
            if not FakeJudgeModule._response_text(record).strip():
                labels[i] = False
            else:
                judge_indices.append(i)

        if not judge_indices:
            return labels, set()

        prompts = [FakeJudgeModule.build_judge_prompt(records[i]) for i in judge_indices]
        raw_outputs = evaluator.predict_batch(prompts, temperature=0.0, max_tokens=16)
        for idx, out in zip(judge_indices, raw_outputs):
            labels[idx] = FakeJudgeModule.parse_judge_output(out)

        retry_indices = [idx for idx in judge_indices if labels[idx] is None]
        retried = set(retry_indices)
        if retry_indices:
            retry_prompts = [FakeJudgeModule.build_judge_prompt(records[i]) for i in retry_indices]
            retry_outputs = evaluator.predict_batch(retry_prompts, temperature=0.0, max_tokens=16)
            for idx, out in zip(retry_indices, retry_outputs):
                labels[idx] = FakeJudgeModule.parse_judge_output(out)

        return labels, retried


class TestLabelRecordsExactMatch:
    def test_correct(self):
        records = [
            {"final": {"answer": "Paris"}, "reference_answer": "paris"},
        ]
        count = label_records(records, FakeExactMatchModule(), evaluator=None)
        assert count == 1
        assert records[0]["final"]["is_correct"] is True
        assert records[0]["final"]["correctness_method"] == "exact_match"

    def test_wrong(self):
        records = [
            {"final": {"answer": "London"}, "reference_answer": "Paris"},
        ]
        label_records(records, FakeExactMatchModule(), evaluator=None)
        assert records[0]["final"]["is_correct"] is False

    def test_only_labels_greedy_records(self):
        records = [
            {"qa_example_id": "0", "final": {"answer": "A"}, "reference_answer": "a"},
            {"qa_example_id": "0", "generation_mode": "sampled", "sample_id": 1, "final": {"answer": "B"}, "reference_answer": "b"},
            {"qa_example_id": "0", "generation_mode": "sampled", "sample_id": 2, "final": {"answer": "C"}, "reference_answer": "c"},
            {"qa_example_id": "1", "final": {"answer": "D"}, "reference_answer": "e"},
        ]
        count = label_records(records, FakeExactMatchModule(), evaluator=None)
        assert count == 2
        assert records[0]["final"]["is_correct"] is True
        assert "is_correct" not in records[1]["final"]
        assert "is_correct" not in records[2]["final"]
        assert records[3]["final"]["is_correct"] is False

    def test_uses_aliases(self):
        records = [
            {"final": {"answer": "NYC"}, "reference_answer": "New York City", "aliases": ["NYC", "New York"]},
        ]
        label_records(records, FakeExactMatchModule(), evaluator=None)
        assert records[0]["final"]["is_correct"] is True

    def test_skips_already_labeled(self):
        records = [
            {"final": {"answer": "Paris", "is_correct": True, "correctness_method": "exact_match"}, "reference_answer": "paris"},
            {"final": {"answer": "London"}, "reference_answer": "London"},
        ]
        count = label_records(records, FakeExactMatchModule(), evaluator=None)
        assert count == 1
        assert records[0]["final"]["is_correct"] is True
        assert records[1]["final"]["is_correct"] is True

    def test_retries_ambiguous_on_rerun(self):
        records = [
            {"final": {"answer": "X", "is_correct": None, "correctness_method": "llm_judge_ambiguous"},
             "reference_answer": "x"},
        ]
        count = label_records(records, FakeExactMatchModule(), evaluator=None)
        assert count == 1
        assert records[0]["final"]["is_correct"] is True

    def test_sampled_records_are_not_labeled(self):
        records = [
            {"qa_example_id": "0", "final": {"answer": "A"}, "reference_answer": "a"},
            {"qa_example_id": "0", "generation_mode": "sampled", "response_sample_id": 2, "sample_id": 5,
             "final": {"answer": "C"}, "reference_answer": "c"},
            {"qa_example_id": "0", "generation_mode": "sampled", "response_sample_id": 1, "sample_id": 4,
             "final": {"answer": "B"}, "reference_answer": "b"},
        ]
        count = label_records(records, FakeExactMatchModule(), evaluator=None)
        assert count == 1
        assert records[0]["final"]["is_correct"] is True
        assert "is_correct" not in records[1]["final"]
        assert "is_correct" not in records[2]["final"]

class TestLabelRecordsLLMJudge:
    def test_batch_judge(self):
        records = [
            {"final": {"answer": "Paris"}, "question": "Capital of France?", "reference_answer": "Paris"},
            {"final": {"answer": "Berlin"}, "question": "Capital of Germany?", "reference_answer": "Berlin"},
        ]
        evaluator = FakeEvaluator(["yes", "no"])
        count = label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert count == 2
        assert records[0]["final"]["is_correct"] is True
        assert records[1]["final"]["is_correct"] is False

    def test_retry_on_ambiguous(self):
        records = [
            {"final": {"answer": "X"}, "question": "Q?", "reference_answer": "X"},
        ]
        evaluator = FakeEvaluator(["unclear", "yes"])
        label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert records[0]["final"]["is_correct"] is True

    def test_double_failure_gives_none(self):
        records = [
            {"final": {"answer": "X"}, "question": "Q?", "reference_answer": "X"},
        ]
        evaluator = FakeEvaluator(["unclear", "still unclear"])
        label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert records[0]["final"]["is_correct"] is None
        assert records[0]["final"]["correctness_method"] == "llm_judge_ambiguous"

    def test_correctness_method_stored(self):
        records = [
            {"final": {"answer": "A"}, "question": "Q?", "reference_answer": "A"},
        ]
        evaluator = FakeEvaluator(["yes"])
        label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert records[0]["final"]["correctness_method"] == "llm_judge"

    def test_retry_correctness_method(self):
        records = [
            {"final": {"answer": "X"}, "question": "Q?", "reference_answer": "X"},
        ]
        evaluator = FakeEvaluator(["unclear", "yes"])
        label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert records[0]["final"]["is_correct"] is True
        assert records[0]["final"]["correctness_method"] == "llm_judge_retry"

    def test_skips_already_labeled_judge(self):
        records = [
            {"final": {"answer": "A", "is_correct": True, "correctness_method": "llm_judge"},
             "question": "Q1?", "reference_answer": "A"},
            {"final": {"answer": "B"}, "question": "Q2?", "reference_answer": "B"},
        ]
        evaluator = FakeEvaluator(["yes"])
        count = label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert count == 1
        assert records[1]["final"]["is_correct"] is True

    def test_empty_response_labeled_incorrect(self):
        records = [
            {"final": {"response": ""}, "question": "Q?", "reference_answer": "A"},
        ]
        evaluator = FakeEvaluator([])
        count = label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert count == 1
        assert records[0]["final"]["is_correct"] is False

    def test_whitespace_response_labeled_incorrect(self):
        records = [
            {"final": {"response": "   "}, "question": "Q?", "reference_answer": "A"},
        ]
        evaluator = FakeEvaluator([])
        count = label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert count == 1
        assert records[0]["final"]["is_correct"] is False

    def test_none_response_labeled_incorrect(self):
        records = [
            {"final": {"response": None}, "question": "Q?", "reference_answer": "A"},
        ]
        evaluator = FakeEvaluator([])
        count = label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert count == 1
        assert records[0]["final"]["is_correct"] is False

    def test_empty_mixed_with_nonempty(self):
        records = [
            {"final": {"answer": "Paris"}, "question": "Q1?", "reference_answer": "Paris"},
            {"final": {"response": ""}, "question": "Q2?", "reference_answer": "X"},
            {"final": {"answer": "Berlin"}, "question": "Q3?", "reference_answer": "Berlin"},
        ]
        evaluator = FakeEvaluator(["yes", "no"])
        count = label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert count == 3
        assert records[0]["final"]["is_correct"] is True
        assert records[1]["final"]["is_correct"] is False
        assert records[2]["final"]["is_correct"] is False

    def test_all_empty_skips_judge(self):
        records = [
            {"final": {"response": ""}, "question": "Q?", "reference_answer": "A"},
            {"final": {"response": "  "}, "question": "Q?", "reference_answer": "B"},
        ]
        evaluator = FakeEvaluator([])
        count = label_records(records, FakeJudgeModule(), evaluator=evaluator)
        assert count == 2
        assert records[0]["final"]["is_correct"] is False
        assert records[1]["final"]["is_correct"] is False


class TestUnlabeledPromptIds:
    def test_all_labeled_returns_empty(self):
        records = [
            {"qa_example_id": "0", "final": {"is_correct": True}},
            {"qa_example_id": "0", "generation_mode": "sampled", "sample_id": 1, "final": {"is_correct": False}},
            {"qa_example_id": "1", "final": {"is_correct": False}},
        ]
        assert unlabeled_prompt_ids(records) == set()

    def test_greedy_ambiguous_excludes_prompt(self):
        records = [
            {"qa_example_id": "0", "final": {"is_correct": None}},
            {"qa_example_id": "0", "generation_mode": "sampled", "sample_id": 1, "final": {"is_correct": True}},
            {"qa_example_id": "1", "final": {"is_correct": True}},
        ]
        assert unlabeled_prompt_ids(records) == {"0"}

    def test_first_sampled_ambiguous_does_not_exclude_prompt(self):
        records = [
            {"qa_example_id": "0", "final": {"is_correct": True}},
            {"qa_example_id": "0", "generation_mode": "sampled", "sample_id": 1, "final": {"is_correct": None}},
            {"qa_example_id": "0", "generation_mode": "sampled", "sample_id": 2, "final": {"is_correct": True}},
            {"qa_example_id": "1", "final": {"is_correct": True}},
        ]
        assert unlabeled_prompt_ids(records) == set()

    def test_missing_is_correct_counts_as_unlabeled(self):
        records = [
            {"qa_example_id": "0", "final": {"answer": "foo"}},
            {"qa_example_id": "1", "final": {"is_correct": True}},
        ]
        assert unlabeled_prompt_ids(records) == {"0"}

    def test_both_prompts_ambiguous(self):
        records = [
            {"qa_example_id": "0", "final": {"is_correct": None}},
            {"qa_example_id": "1", "final": {"is_correct": None}},
        ]
        assert unlabeled_prompt_ids(records) == {"0", "1"}

    def test_empty_records(self):
        assert unlabeled_prompt_ids([]) == set()

    def test_second_sampled_ambiguous_does_not_exclude(self):
        records = [
            {"qa_example_id": "0", "final": {"is_correct": True}},
            {"qa_example_id": "0", "generation_mode": "sampled", "sample_id": 1, "final": {"is_correct": True}},
            {"qa_example_id": "0", "generation_mode": "sampled", "sample_id": 2, "final": {"is_correct": None}},
        ]
        assert unlabeled_prompt_ids(records) == set()
