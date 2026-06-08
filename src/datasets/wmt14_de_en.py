"""WMT14 German-English dataset adapter."""

from __future__ import annotations

from src.datasets.wmt_translation import make_translation_adapter

DATASET_NAME = "wmt14_de_en"
HF_NAME = "wmt/wmt14"
DEFAULT_CONFIG_NAME = "de-en"
DEFAULT_SPLIT = "test"
SOURCE_LANGUAGE = "de"
TARGET_LANGUAGE = "en"

_adapter = make_translation_adapter(dict(
    dataset_name=DATASET_NAME,
    hf_name=HF_NAME,
    default_config_name=DEFAULT_CONFIG_NAME,
    default_split=DEFAULT_SPLIT,
    source_language=SOURCE_LANGUAGE,
    target_language=TARGET_LANGUAGE,
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
