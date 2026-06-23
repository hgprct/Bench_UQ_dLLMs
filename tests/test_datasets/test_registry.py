"""Tests for the dataset adapter registry."""

import inspect

import pytest

from src.config import Dataset
from src.registry import get_dataset_module


class TestGetDatasetModule:
    @pytest.mark.parametrize("dataset", list(Dataset), ids=[d.value for d in Dataset])
    def test_every_dataset_resolves_with_uniform_signature(self, dataset):
        module = get_dataset_module(dataset.value)
        # Every adapter exposes extract_qa_sample(example, id) and format_prompt.
        assert hasattr(module, "extract_qa_sample")
        assert hasattr(module, "format_prompt")
        params = list(inspect.signature(module.extract_qa_sample).parameters)
        assert params[:2] == ["example", "id"], (dataset.value, params)

    def test_unknown_dataset_raises_valueerror(self):
        with pytest.raises(ValueError):
            get_dataset_module("not_a_dataset")
