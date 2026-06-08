"""XSum extreme summarization dataset adapter."""

from __future__ import annotations

from src.datasets.summarization import make_summarization_adapter

DATASET_NAME = "xsum"
HF_NAME = "EdinburghNLP/xsum"
DEFAULT_CONFIG_NAME = None
DEFAULT_SPLIT = "test"
SOURCE_FIELD = "document"
SUMMARY_FIELD = "summary"

_adapter = make_summarization_adapter(dict(
    dataset_name=DATASET_NAME,
    hf_name=HF_NAME,
    default_config_name=DEFAULT_CONFIG_NAME,
    default_split=DEFAULT_SPLIT,
    source_field=SOURCE_FIELD,
    summary_field=SUMMARY_FIELD,
))

load_dataset = _adapter.load_dataset
extract_question_and_answer = _adapter.extract_question_and_answer
format_prompt = _adapter.format_prompt
few_shot_prefix = _adapter.few_shot_prefix
parse_response = _adapter.parse_response
extract_short_answer = _adapter.extract_short_answer
normalize_answer = _adapter.normalize_answer
cluster_key = _adapter.cluster_key
build_judge_prompt = _adapter.build_judge_prompt
parse_judge_output = _adapter.parse_judge_output
