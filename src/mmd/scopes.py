"""Step scope definitions and selection for trajectory analysis."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class StepScope:
    """Defines which denoising steps to select from a trajectory."""
    name: str
    mode: str
    k: int | None = None
    range_index: int | None = None
    num_ranges: int | None = None


DEFAULT_SCOPE_K = 8
DEFAULT_NUM_RANGES = 5


def build_scopes(k: int, num_ranges: int) -> list[StepScope]:
    """Build the canonical list of step scopes: full + random + ranges."""
    scopes = [
        StepScope(name="full", mode="full"),
        StepScope(name=f"random:{k}", mode="random", k=k),
    ]
    for i in range(num_ranges):
        scopes.append(StepScope(
            name=f"range-{i}", mode="range",
            k=k, range_index=i, num_ranges=num_ranges,
        ))
    return scopes


def select_steps(
    num_steps: int,
    mode: str,
    *,
    k: int | None = None,
    range_index: int | None = None,
    num_ranges: int | None = None,
    rng: random.Random | None = None,
) -> list[int]:
    """Select step indices according to the scope mode."""
    num_steps = int(num_steps)
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}")
    if mode in ("all", "full"):
        return list(range(num_steps))
    if k is None:
        raise ValueError(f"Step mode '{mode}' requires k")
    k = min(int(k), num_steps)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if mode == "random":
        rng = rng or random.Random()
        return sorted(rng.sample(range(num_steps), k))
    if mode == "range":
        if range_index is None or num_ranges is None:
            raise ValueError("Range mode requires range_index and num_ranges")
        if num_steps <= k:
            return list(range(num_steps))
        offset = _range_offset(int(range_index), int(num_ranges), k, num_steps)
        return list(range(offset, offset + k))
    raise ValueError(f"Unsupported step mode '{mode}'. Use full, random, or range.")


def _range_offset(range_index: int, num_ranges: int, k: int, num_steps: int) -> int:
    max_start = max(0, num_steps - k)
    if num_ranges <= 1:
        return 0
    return int(round(range_index * max_start / (num_ranges - 1)))
