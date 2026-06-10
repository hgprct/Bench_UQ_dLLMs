"""Tests for src.config -- enums, defaults, run_id, config building."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    DATASET_CONFIGS,
    MODEL_HF_IDS,
    REMASKING_FULL_NAMES,
    Dataset,
    Model,
    Remasking,
    build_generation_config,
    config_filename,
    load_config,
    run_id,
    save_config,
)


class TestEnums:
    def test_model_values(self):
        assert Model.LLaDA.value == "LLaDA"
        assert Model.LLaDA15.value == "LLaDA1.5"

    def test_dataset_values(self):
        assert Dataset.triviaqa.value == "triviaqa"
        assert Dataset.gsm8k.value == "gsm8k"
        assert Dataset.wmt14_fr_en.value == "wmt14_fr_en"
        assert Dataset.xsum.value == "xsum"
        assert Dataset.samsum.value == "samsum"
        assert Dataset.hotpotqa.value == "hotpotqa"
        assert Dataset.musique.value == "musique"

    def test_remasking_values(self):
        assert Remasking.lc.value == "lc"
        assert Remasking.rd.value == "rd"

    def test_all_models_have_hf_ids(self):
        for model in Model:
            assert model in MODEL_HF_IDS

    def test_all_datasets_have_configs(self):
        for dataset in Dataset:
            assert dataset in DATASET_CONFIGS

    def test_all_remaskings_have_full_names(self):
        for remasking in Remasking:
            assert remasking in REMASKING_FULL_NAMES

    def test_invalid_model_raises(self):
        try:
            Model("invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_dataset_raises(self):
        try:
            Dataset("invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestRunId:
    def test_basic(self):
        result = run_id(Model.LLaDA, Dataset.triviaqa, 128, 64, Remasking.lc)
        assert result == "LLaDA_triviaqa_l128_s64_lc"

    def test_with_block_size(self):
        result = run_id(Model.LLaDA, Dataset.triviaqa, 128, 64, Remasking.lc, block_size=32)
        assert result == "LLaDA_triviaqa_l128_s64_lc_b32"

    def test_with_temperature(self):
        result = run_id(Model.LLaDA15, Dataset.gsm8k, 64, 32, Remasking.rd, temperature=0.7)
        assert result == "LLaDA1.5_gsm8k_l64_s32_rd_t0p7"

    def test_temperature_negative(self):
        result = run_id(Model.LLaDA15, Dataset.wmt14_fr_en, 128, 64, Remasking.lc, temperature=-0.5)
        assert "m0p5" in result

    def test_all_combinations_unique(self):
        ids = set()
        for model in Model:
            for dataset in Dataset:
                for remasking in Remasking:
                    rid = run_id(model, dataset, 128, 64, remasking)
                    assert rid not in ids, f"Duplicate run_id: {rid}"
                    ids.add(rid)


class TestConfigFilename:
    def test_basic(self):
        result = config_filename(Model.LLaDA, Dataset.triviaqa, 128, 64, Remasking.lc)
        assert result == "LLaDA_triviaqa_l128_s64_lc.json"

    def test_with_block_size(self):
        result = config_filename(Model.LLaDA, Dataset.triviaqa, 128, 64, Remasking.lc, block_size=32)
        assert result == "LLaDA_triviaqa_l128_s64_lc_b32.json"


class TestBuildGenerationConfig:
    def test_basic_structure(self):
        config = build_generation_config(
            Model.LLaDA, Dataset.triviaqa, 128, 64, Remasking.lc,
        )
        assert config["model_id"] == MODEL_HF_IDS[Model.LLaDA]
        assert config["dataset"] == "triviaqa"
        assert config["max_gen_length"] == 128
        assert config["steps"] == 64
        assert config["remasking"] == "lc"
        assert config["num_questions"] == 1000
        assert config["temperature"] == 1.0
        assert config["generate_greedy"] is True

    def test_block_size_from_dataset(self):
        config = build_generation_config(Model.LLaDA, Dataset.triviaqa)
        ds_config = DATASET_CONFIGS[Dataset.triviaqa]
        assert config["block_size"] == ds_config.block_size

    def test_block_size_explicit(self):
        config = build_generation_config(Model.LLaDA, Dataset.triviaqa, block_size=32)
        assert config["block_size"] == 32

    def test_block_size_none_overrides_dataset(self):
        config = build_generation_config(Model.LLaDA, Dataset.triviaqa, block_size=None)
        assert config["block_size"] is None

    def test_generate_greedy_false(self):
        config = build_generation_config(
            Model.LLaDA, Dataset.triviaqa, 128, 64, Remasking.lc,
            generate_greedy=False,
        )
        assert config["generate_greedy"] is False

    def test_overrides(self):
        config = build_generation_config(
            Model.LLaDA, Dataset.triviaqa, 128, 64, Remasking.lc,
            num_questions=50, batch_size=4,
        )
        assert config["num_questions"] == 50
        assert config["batch_size"] == 4

    def test_dataset_defaults_applied(self):
        config = build_generation_config(
            Model.LLaDA, Dataset.gsm8k, 128, 64, Remasking.lc,
        )
        ds_config = DATASET_CONFIGS[Dataset.gsm8k]
        assert config["dataset_config_name"] == ds_config.config_name
        assert config["split"] == ds_config.split
        assert config["batch_size"] == ds_config.batch_size

    def test_per_dataset_gen_length_steps(self):
        config = build_generation_config(Model.LLaDA, Dataset.gsm8k)
        ds_config = DATASET_CONFIGS[Dataset.gsm8k]
        assert config["max_gen_length"] == ds_config.max_gen_length
        assert config["steps"] == ds_config.steps

    def test_explicit_length_overrides_dataset_default(self):
        config = build_generation_config(Model.LLaDA, Dataset.gsm8k, 64, 32)
        assert config["max_gen_length"] == 64
        assert config["steps"] == 32


class TestSaveLoadConfig:
    def test_roundtrip(self, tmp_path):
        config = {"model_id": "test", "steps": 32}
        path = tmp_path / "test.json"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded == config

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "config.json"
        save_config({"test": True}, path)
        assert path.exists()
        with open(path) as f:
            assert json.load(f) == {"test": True}
