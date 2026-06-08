"""MMD^2 estimators: biased and unbiased."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KernelBlocks:
    """Kernel matrices needed by an empirical two-sample MMD estimate."""
    xx: np.ndarray
    yy: np.ndarray
    xy: np.ndarray


def mmd2_biased(blocks: KernelBlocks) -> float:
    """Biased MMD^2 estimate: E[k(X,X)] + E[k(Y,Y)] - 2*E[k(X,Y)]."""
    xx = np.asarray(blocks.xx, dtype=np.float64)
    yy = np.asarray(blocks.yy, dtype=np.float64)
    xy = np.asarray(blocks.xy, dtype=np.float64)
    if xx.size == 0 or yy.size == 0 or xy.size == 0:
        return float("nan")
    value = float(xx.mean() + yy.mean() - 2.0 * xy.mean())
    if -1e-10 < value < 0.0:
        return 0.0
    return value


def mmd2_unbiased(
    blocks: KernelBlocks,
    traj_ids_x: np.ndarray,
    traj_ids_y: np.ndarray,
) -> float:
    """Unbiased MMD^2 estimate excluding within-trajectory pairs."""
    xx = np.asarray(blocks.xx, dtype=np.float64)
    yy = np.asarray(blocks.yy, dtype=np.float64)
    xy = np.asarray(blocks.xy, dtype=np.float64)

    n_y = yy.shape[0]

    mask_xx = traj_ids_x[:, None] != traj_ids_x[None, :]
    n_valid_xx = mask_xx.sum()
    if n_valid_xx == 0:
        return float("nan")
    xx_term = (xx * mask_xx).sum() / n_valid_xx

    if n_y < 2:
        return float("nan")
    yy_term = (yy.sum() - np.trace(yy)) / (n_y * (n_y - 1))

    mask_xy = traj_ids_x[:, None] != traj_ids_y[None, :]
    n_valid_xy = mask_xy.sum()
    if n_valid_xy == 0:
        return float("nan")
    xy_term = (xy * mask_xy).sum() / n_valid_xy

    return float(xx_term + yy_term - 2.0 * xy_term)


def estimate_mmd2(
    blocks: KernelBlocks,
    estimator: str,
    *,
    traj_ids_x: np.ndarray | None = None,
    traj_ids_y: np.ndarray | None = None,
) -> float:
    """Dispatch to biased or unbiased MMD^2 estimator."""
    if estimator == "biased":
        return mmd2_biased(blocks)
    if estimator == "unbiased":
        if traj_ids_x is None or traj_ids_y is None:
            raise ValueError("traj_ids_x and traj_ids_y are required for the unbiased estimator")
        return mmd2_unbiased(blocks, traj_ids_x, traj_ids_y)
    raise ValueError(f"Unsupported estimator '{estimator}'. Use biased or unbiased.")


def finite_mean_std(values: list[float]) -> tuple[float, float, int]:
    """Mean and std of finite values, returning (mean, std, count)."""
    finite = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return float(finite.mean()), std, int(finite.size)
