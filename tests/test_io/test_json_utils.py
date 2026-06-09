"""Tests for src.io.json_utils."""

import numpy as np

from src.utils.io import read_json, read_jsonl, to_jsonable, write_json, write_jsonl


class TestToJsonable:
    def test_numpy_scalar(self):
        assert to_jsonable(np.float64(3.14)) == 3.14
        assert to_jsonable(np.int64(42)) == 42

    def test_numpy_array(self):
        result = to_jsonable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        result = to_jsonable({"a": np.float32(1.0), "b": [np.int32(2)]})
        assert result == {"a": 1.0, "b": [2]}

    def test_passthrough(self):
        assert to_jsonable("hello") == "hello"
        assert to_jsonable(42) == 42


class TestReadWriteJson:
    def test_roundtrip(self, tmp_path):
        data = {"model": "test", "steps": 32, "values": [1.0, 2.0]}
        path = tmp_path / "test.json"
        write_json(path, data)
        assert read_json(path) == data

    def test_numpy_in_values(self, tmp_path):
        data = {"score": np.float64(0.95), "counts": np.array([1, 2, 3])}
        path = tmp_path / "test.json"
        write_json(path, data)
        loaded = read_json(path)
        assert loaded["score"] == 0.95
        assert loaded["counts"] == [1, 2, 3]


class TestReadWriteJsonl:
    def test_roundtrip(self, tmp_path):
        rows = [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"}]
        path = tmp_path / "test.jsonl"
        write_jsonl(path, rows)
        loaded = read_jsonl(path)
        assert loaded == rows

    def test_empty_lines_skipped(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n')
        loaded = read_jsonl(path)
        assert len(loaded) == 2
