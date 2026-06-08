"""Deterministic train/val/test splitting at the prompt level."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def split_prompts(
    prompt_ids: list[str],
    *,
    train: int = 10,
    val: int = 30,
    test: int = 40,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split prompt IDs into train/val/test sets using absolute counts.

    Args:
        prompt_ids: All prompt IDs to split.
        train: Number of prompts for training (step selection QP).
        val: Number of prompts for validation (lambda tuning).
        test: Number of prompts for test evaluation.
        seed: Random seed for reproducible shuffling.

    Returns:
        Dict with keys 'train', 'val', 'test', each containing a sorted
        list of prompt IDs.

    Raises:
        ValueError: If requested counts exceed available prompts.
    """
    total_requested = train + val + test
    if total_requested > len(prompt_ids):
        raise ValueError(
            f"Requested {total_requested} prompts (train={train}, val={val}, test={test}) "
            f"but only {len(prompt_ids)} available"
        )

    rng = random.Random(seed)
    shuffled = list(prompt_ids)
    rng.shuffle(shuffled)

    return {
        "train": sorted(shuffled[:train]),
        "val": sorted(shuffled[train:train + val]),
        "test": sorted(shuffled[train + val:train + val + test]),
    }


def save_splits(
    splits: dict[str, list[str]],
    path: str | Path,
    *,
    seed: int = 42,
    train: int = 10,
    val: int = 30,
    test: int = 40,
) -> None:
    """Write splits.json sidecar file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "seed": seed,
        "counts": {"train": train, "val": val, "test": test},
        "prompt_ids": splits,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """Load splits.json and return the prompt_ids dict."""
    with open(path) as f:
        data = json.load(f)
    return data["prompt_ids"]


def _stratified_blocks(
    ids: list[str],
    labels: dict[str, bool],
    k: int,
    rng: random.Random,
) -> list[list[str]]:
    """Partition *ids* into *k* blocks with proportional class balance."""
    pos = [pid for pid in ids if labels.get(pid, True)]
    neg = [pid for pid in ids if not labels.get(pid, True)]
    rng_copy = random.Random(rng.random())
    rng_copy.shuffle(pos)
    rng_copy2 = random.Random(rng.random())
    rng_copy2.shuffle(neg)

    def _split_list(lst: list[str]) -> list[list[str]]:
        base = len(lst) // k
        rem = len(lst) % k
        blocks: list[list[str]] = []
        off = 0
        for i in range(k):
            sz = base + (1 if i < rem else 0)
            blocks.append(lst[off:off + sz])
            off += sz
        return blocks

    pos_blocks = _split_list(pos)
    neg_blocks = _split_list(neg)
    return [pos_blocks[i] + neg_blocks[i] for i in range(k)]


def _stratified_train_val(
    remaining: list[str],
    labels: dict[str, bool],
    n_train: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Split remaining into train/val preserving class ratio."""
    pos = [pid for pid in remaining if labels.get(pid, True)]
    neg = [pid for pid in remaining if not labels.get(pid, True)]
    rng_copy = random.Random(rng.random())
    rng_copy.shuffle(pos)
    rng_copy2 = random.Random(rng.random())
    rng_copy2.shuffle(neg)

    n_neg_train = round(n_train * len(neg) / max(len(remaining), 1))
    n_neg_train = max(min(n_neg_train, len(neg)), 0)
    n_pos_train = min(n_train - n_neg_train, len(pos))

    train = pos[:n_pos_train] + neg[:n_neg_train]
    val = pos[n_pos_train:] + neg[n_neg_train:]
    return train, val


def build_kfold_splits(
    prompt_ids: list[str],
    *,
    k: int = 5,
    n_train: int,
    seed: int = 42,
    labels: dict[str, bool] | None = None,
) -> list[dict[str, list[str]]]:
    """Build K non-overlapping test folds with fixed train size.

    Test partitions cover all prompt_ids exactly once.  Within each fold
    the remaining prompts are split into train (absolute count given by
    *n_train*) and val (everything left).

    When *labels* is provided (prompt_id → is_correct), all splits are
    stratified so that each fold's test, train, and val preserve the
    overall class ratio.

    Raises ValueError if n_train exceeds the non-test budget of any fold.
    """
    n = len(prompt_ids)
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds the number of prompts ({n})")

    rng = random.Random(seed)
    shuffled = list(prompt_ids)
    rng.shuffle(shuffled)

    if labels is not None:
        blocks = _stratified_blocks(shuffled, labels, k, rng)
    else:
        test_base = n // k
        remainder = n % k
        blocks = []
        offset = 0
        for i in range(k):
            block_size = test_base + (1 if i < remainder else 0)
            blocks.append(shuffled[offset:offset + block_size])
            offset += block_size

    folds: list[dict[str, list[str]]] = []
    for fold_idx in range(k):
        test_ids = blocks[fold_idx]
        test_set = set(test_ids)
        remaining = [pid for pid in shuffled if pid not in test_set]

        if n_train > len(remaining):
            raise ValueError(
                f"n_train={n_train} exceeds available non-test prompts "
                f"({len(remaining)}) in fold {fold_idx} "
                f"(total={n}, test_size={len(test_ids)})"
            )

        fold_rng = random.Random(seed + fold_idx + 1)

        if labels is not None:
            train_ids, val_ids = _stratified_train_val(remaining, labels, n_train, fold_rng)
        else:
            fold_remaining = list(remaining)
            fold_rng.shuffle(fold_remaining)
            train_ids = fold_remaining[:n_train]
            val_ids = fold_remaining[n_train:]

        folds.append({
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        })

    return folds


def save_kfold_splits(
    folds: list[dict[str, list[str]]],
    path: str | Path,
    *,
    k: int,
    n_train: int,
    seed: int,
) -> None:
    """Write kfold_splits.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "k": k,
        "n_train": n_train,
        "seed": seed,
        "folds": [
            {
                "fold": i,
                "counts": {
                    "train": len(f["train"]),
                    "val": len(f["val"]),
                    "test": len(f["test"]),
                },
                "prompt_ids": f,
            }
            for i, f in enumerate(folds)
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_kfold_splits(path: str | Path) -> list[dict[str, list[str]]]:
    """Load kfold_splits.json and return list of fold dicts."""
    with open(path) as f:
        data = json.load(f)
    return [fold["prompt_ids"] for fold in data["folds"]]
