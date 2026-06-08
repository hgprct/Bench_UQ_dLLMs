"""Tests for src.split -- deterministic train/val/test splitting."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from split import (
    build_kfold_splits,
    load_kfold_splits,
    load_splits,
    save_kfold_splits,
    save_splits,
    split_prompts,
)


class TestSplitPrompts:
    def test_basic_counts(self):
        ids = [str(i) for i in range(100)]
        splits = split_prompts(ids, train=10, val=30, test=40)
        assert len(splits["train"]) == 10
        assert len(splits["val"]) == 30
        assert len(splits["test"]) == 40

    def test_disjoint(self):
        ids = [str(i) for i in range(100)]
        splits = split_prompts(ids, train=10, val=30, test=40)
        all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert len(all_assigned) == 80

    def test_no_overlap(self):
        ids = [str(i) for i in range(100)]
        splits = split_prompts(ids, train=10, val=30, test=40)
        train = set(splits["train"])
        val = set(splits["val"])
        test = set(splits["test"])
        assert not train & val
        assert not train & test
        assert not val & test

    def test_deterministic(self):
        ids = [str(i) for i in range(100)]
        a = split_prompts(ids, train=10, val=30, test=40, seed=42)
        b = split_prompts(ids, train=10, val=30, test=40, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        ids = [str(i) for i in range(100)]
        a = split_prompts(ids, train=10, val=30, test=40, seed=42)
        b = split_prompts(ids, train=10, val=30, test=40, seed=99)
        assert a != b

    def test_sorted_output(self):
        ids = [str(i) for i in range(100)]
        splits = split_prompts(ids, train=10, val=30, test=40)
        for key in ("train", "val", "test"):
            assert splits[key] == sorted(splits[key])

    def test_exceeds_available_raises(self):
        ids = [str(i) for i in range(10)]
        try:
            split_prompts(ids, train=5, val=5, test=5)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "15" in str(e) and "10" in str(e)

    def test_exact_fit(self):
        ids = [str(i) for i in range(80)]
        splits = split_prompts(ids, train=10, val=30, test=40)
        all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert len(all_assigned) == 80

    def test_all_ids_from_input(self):
        ids = [str(i) for i in range(100)]
        splits = split_prompts(ids, train=10, val=30, test=40)
        all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert all_assigned.issubset(set(ids))


class TestBuildKfoldSplits:
    def test_test_folds_cover_all_prompts(self):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        all_test = []
        for f in folds:
            all_test.extend(f["test"])
        assert sorted(all_test) == sorted(ids)

    def test_test_folds_non_overlapping(self):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        seen = set()
        for f in folds:
            test_set = set(f["test"])
            assert not seen & test_set, "Test folds overlap"
            seen |= test_set

    def test_train_size_matches(self):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        for f in folds:
            assert len(f["train"]) == 10

    def test_val_is_remainder(self):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        for f in folds:
            expected_val = len(ids) - len(f["test"]) - 10
            assert len(f["val"]) == expected_val

    def test_no_overlap_within_fold(self):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        for f in folds:
            train = set(f["train"])
            val = set(f["val"])
            test = set(f["test"])
            assert not train & val
            assert not train & test
            assert not val & test

    def test_all_ids_from_input(self):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        input_set = set(ids)
        for f in folds:
            assert set(f["train"]) | set(f["val"]) | set(f["test"]) == input_set

    def test_remainder_distribution(self):
        ids = [str(i) for i in range(83)]
        folds = build_kfold_splits(ids, k=5, n_train=5, seed=42)
        test_sizes = [len(f["test"]) for f in folds]
        assert sum(test_sizes) == 83
        assert max(test_sizes) - min(test_sizes) <= 1
        assert test_sizes.count(17) == 3
        assert test_sizes.count(16) == 2

    def test_deterministic(self):
        ids = [str(i) for i in range(80)]
        a = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        b = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        ids = [str(i) for i in range(80)]
        a = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        b = build_kfold_splits(ids, k=5, n_train=10, seed=99)
        assert a != b

    def test_sorted_output(self):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        for f in folds:
            for key in ("train", "val", "test"):
                assert f[key] == sorted(f[key])

    def test_train_exceeds_budget_raises(self):
        ids = [str(i) for i in range(20)]
        try:
            build_kfold_splits(ids, k=5, n_train=17, seed=42)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "n_train=17" in str(e)

    def test_k_too_small_raises(self):
        ids = [str(i) for i in range(10)]
        try:
            build_kfold_splits(ids, k=1, n_train=2, seed=42)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_k_exceeds_n_raises(self):
        ids = [str(i) for i in range(3)]
        try:
            build_kfold_splits(ids, k=5, n_train=1, seed=42)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_small_dataset_works(self):
        """Datasets like samsum may have very few prompts after filtering."""
        ids = [str(i) for i in range(15)]
        folds = build_kfold_splits(ids, k=5, n_train=2, seed=42)
        assert len(folds) == 5
        for f in folds:
            assert len(f["train"]) == 2
            assert len(f["train"]) + len(f["val"]) + len(f["test"]) == 15

    def test_minimal_viable_folds(self):
        """Minimum: k=2, n_train=1, 3 prompts."""
        ids = [str(i) for i in range(3)]
        folds = build_kfold_splits(ids, k=2, n_train=1, seed=42)
        assert len(folds) == 2
        all_test = []
        for f in folds:
            all_test.extend(f["test"])
            assert len(f["train"]) == 1
        assert sorted(all_test) == sorted(ids)


class TestStratifiedKfoldSplits:
    def test_class_balance_preserved(self):
        ids = [str(i) for i in range(1000)]
        labels = {str(i): i < 908 for i in range(1000)}
        folds = build_kfold_splits(ids, k=3, n_train=100, seed=42, labels=labels)
        overall_rate = 92 / 1000
        for fold in folds:
            for split_name in ("test", "train", "val"):
                split_ids = fold[split_name]
                n_neg = sum(1 for pid in split_ids if not labels[pid])
                rate = n_neg / len(split_ids)
                assert abs(rate - overall_rate) < 0.02, (
                    f"{split_name}: minority rate {rate:.3f} deviates from {overall_rate:.3f}"
                )

    def test_full_coverage_and_disjoint(self):
        ids = [str(i) for i in range(200)]
        labels = {str(i): i < 180 for i in range(200)}
        folds = build_kfold_splits(ids, k=5, n_train=20, seed=42, labels=labels)
        all_test = []
        for fold in folds:
            s_tr, s_va, s_te = set(fold["train"]), set(fold["val"]), set(fold["test"])
            assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
            assert len(s_tr) + len(s_va) + len(s_te) == 200
            all_test.extend(fold["test"])
        assert sorted(all_test) == sorted(ids)

    def test_labels_none_matches_default(self):
        ids = [str(i) for i in range(80)]
        a = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        b = build_kfold_splits(ids, k=5, n_train=10, seed=42, labels=None)
        assert a == b


class TestSaveLoadKfoldSplits:
    def test_roundtrip(self, tmp_path):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        path = tmp_path / "kfold_splits.json"
        save_kfold_splits(folds, path, k=5, n_train=10, seed=42)
        loaded = load_kfold_splits(path)
        assert loaded == folds

    def test_metadata_saved(self, tmp_path):
        ids = [str(i) for i in range(80)]
        folds = build_kfold_splits(ids, k=5, n_train=10, seed=42)
        path = tmp_path / "kfold_splits.json"
        save_kfold_splits(folds, path, k=5, n_train=10, seed=42)
        with open(path) as f:
            data = json.load(f)
        assert data["k"] == 5
        assert data["n_train"] == 10
        assert data["seed"] == 42
        assert len(data["folds"]) == 5


class TestSaveLoadSplits:
    def test_roundtrip(self, tmp_path):
        ids = [str(i) for i in range(100)]
        splits = split_prompts(ids, train=10, val=30, test=40, seed=42)
        path = tmp_path / "splits.json"
        save_splits(splits, path, seed=42, train=10, val=30, test=40)
        loaded = load_splits(path)
        assert loaded == splits

    def test_metadata_saved(self, tmp_path):
        ids = [str(i) for i in range(100)]
        splits = split_prompts(ids, train=10, val=30, test=40, seed=42)
        path = tmp_path / "splits.json"
        save_splits(splits, path, seed=42, train=10, val=30, test=40)
        with open(path) as f:
            data = json.load(f)
        assert data["seed"] == 42
        assert data["counts"] == {"train": 10, "val": 30, "test": 40}
