"""CUDA-sync wall-clock timer + warmup helpers for latency profiling."""

from __future__ import annotations

import time
from typing import Any


class Timer:
    """Context manager: wall-clock time with optional CUDA sync at entry and exit."""

    def __init__(self, device: Any = None, *, sync: bool = True) -> None:
        self.device = device
        self.sync = sync
        self.elapsed: float | None = None
        self._t0: float = 0.0

    def _maybe_sync(self) -> None:
        if not self.sync:
            return
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def __enter__(self) -> "Timer":
        self._maybe_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._maybe_sync()
        self.elapsed = time.perf_counter() - self._t0


def autodetect_nli_batch(
    nli_model: Any,
    device: Any,
    *,
    candidates: tuple[int, ...] = (32, 64, 128, 256),
    dummy_seq_len: int = 64,
) -> int:
    """Find the largest NLI batch size that fits in VRAM via a ladder probe.

    The probe runs once at warmup and locks the value for the rest of the run.
    Falls back to the smallest candidate on the first OOM.
    """
    import torch

    largest_ok: int | None = None
    dummy_word = "lorem " * (dummy_seq_len // 6 + 1)
    dummy_p = [dummy_word] * candidates[-1]
    dummy_h = [dummy_word] * candidates[-1]

    for bs in candidates:
        try:
            nli_model.batch_probabilities(dummy_p[:bs], dummy_h[:bs], batch_size=bs)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            largest_ok = bs
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    if largest_ok is None:
        raise RuntimeError(
            f"NLI model OOMs at smallest candidate batch_size={candidates[0]}; "
            "reduce candidates or free GPU memory"
        )
    return largest_ok


def warmup_nli(nli_model: Any, device: Any, *, batch_size: int, n_calls: int = 2) -> None:
    """Run dummy NLI batches so first real timing isn't skewed by lazy kernel compilation."""
    import torch

    dummy = ["The quick brown fox jumps over the lazy dog."] * batch_size
    for _ in range(n_calls):
        nli_model.batch_probabilities(dummy, dummy, batch_size=batch_size)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
