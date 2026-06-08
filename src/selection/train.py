"""Learn continuous temporal weights via capped simplex QP."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.io.json_utils import to_jsonable
from src.mmd.kernels import EmbeddingKernel
from src.mmd.load import PromptTrajectories, is_valid_text, load_generated_run


def step_embeddings(
    prompt: PromptTrajectories,
    embedder,
    *,
    num_samples: int | None = None,
    exclude_greedy: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Embed all step answers for a prompt. Returns (embeddings, N, T) with shape (N, T, D)."""
    samples = sorted(prompt.samples, key=lambda s: s.response_sample_id)
    if exclude_greedy:
        filtered = [s for s in samples if s.response_sample_id > 0]
        if filtered:
            samples = filtered
    if num_samples is not None:
        samples = samples[:num_samples]
    N = len(samples)
    T = prompt.num_steps

    valid_texts: list[str] = []
    valid_positions: list[tuple[int, int]] = []
    for n, sample in enumerate(samples):
        for t in range(T):
            if t < len(sample.step_answers) and is_valid_text(sample.step_answers[t]):
                valid_texts.append(sample.step_answers[t].strip())
                valid_positions.append((n, t))

    if not valid_texts:
        return np.empty((0, 0, 0), dtype=np.float32), N, T

    valid_embs = embedder.encode(valid_texts)
    D = valid_embs.shape[1]
    emb_matrix = np.zeros((N, T, D), dtype=np.float32)
    for pos, (n, t) in enumerate(valid_positions):
        emb_matrix[n, t] = valid_embs[pos]

    return emb_matrix, N, T


def compute_kernel_matrices(
    emb: np.ndarray,
    kernel: EmbeddingKernel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute A^V, A^U, and C matrices for one prompt. Returns (A_V, A_U, C)."""
    N, T, _D = emb.shape
    if N < 2 or T == 0:
        return (
            np.zeros((T, T), dtype=np.float64),
            np.zeros((T, T), dtype=np.float64),
            np.zeros((T, T), dtype=np.float64),
        )

    # Resolve bandwidth once from ALL embeddings for PSD guarantees
    sigma_sq = kernel.resolve_sigma_sq(emb.reshape(-1, emb.shape[-1]))

    A_V = np.zeros((T, T), dtype=np.float64)
    A_U = np.zeros((T, T), dtype=np.float64)
    C = np.zeros((T, T), dtype=np.float64)

    for t in range(T):
        for u in range(t, T):
            K = np.asarray(
                kernel.cross_kernel(emb[:, t, :], emb[:, u, :], sigma_sq=sigma_sq),
                dtype=np.float64,
            )
            A_V[t, u] = K.mean()
            A_V[u, t] = A_V[t, u]

            off_diag_sum = K.sum() - np.trace(K)
            A_U[t, u] = off_diag_sum / (N * (N - 1))
            A_U[u, t] = A_U[t, u]

            C[t, u] = np.trace(K) / N
            C[u, t] = C[t, u]

    return A_V, A_U, C


def compute_delta_U(C: np.ndarray, A_U: np.ndarray) -> np.ndarray:
    return C - A_U


def verify_psd(matrix: np.ndarray, name: str, tol: float = -1e-8) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    min_eig = float(eigenvalues[0])
    max_eig = float(eigenvalues[-1])
    return {
        "name": name,
        "shape": list(matrix.shape),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "is_psd": min_eig >= tol,
        "eigenvalues": eigenvalues.tolist(),
    }


def _project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection of v onto {w : 0 <= w_i <= cap, sum(w) = 1}."""
    n = v.size
    if n * cap < 1.0 - 1e-12:
        raise ValueError(f"Infeasible capped simplex: n*cap={n*cap} < 1")
    lo = float(v.min()) - 1.0
    hi = float(v.max())
    for _ in range(100):
        tau = 0.5 * (lo + hi)
        s = np.clip(v - tau, 0.0, cap).sum()
        if abs(s - 1.0) < 1e-12:
            break
        if s > 1.0:
            lo = tau
        else:
            hi = tau
    return np.clip(v - tau, 0.0, cap)


def solve_qp(
    A_bar: np.ndarray,
    b_bar: np.ndarray,
    Delta_bar: np.ndarray,
    c_bar: float,
    *,
    lambda_reg: float,
    budget_k: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the capped simplex QP for temporal weight learning."""
    import cvxpy as cp

    T = A_bar.shape[0]
    if budget_k > T:
        raise ValueError(
            f"budget_k={budget_k} exceeds T={T}: the capped simplex is infeasible."
        )
    diagnostics: dict[str, Any] = {"solver": "cvxpy", "T": T, "budget_k": budget_k, "lambda_reg": lambda_reg}

    Q = A_bar + lambda_reg * Delta_bar
    Q = 0.5 * (Q + Q.T)

    min_eig = float(np.linalg.eigvalsh(Q).min())
    if min_eig < 0:
        shift = -min_eig + 1e-8
        Q += shift * np.eye(T)
        diagnostics["psd_correction"] = float(shift)

    w = cp.Variable(T)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Q)) - 2.0 * b_bar @ w)

    constraints = [
        w >= 0,
        w <= 1.0 / budget_k,
        cp.sum(w) == 1.0,
    ]

    problem = cp.Problem(objective, constraints)

    t0 = time.time()
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except cp.error.SolverError:
        pass

    diagnostics["status"] = problem.status
    diagnostics["optimal_value"] = float(problem.value) + c_bar if problem.value is not None else None

    if w.value is None:
        problem.solve(solver=cp.SCS, verbose=False, max_iters=10000)
        diagnostics["status"] = problem.status
        diagnostics["optimal_value"] = float(problem.value) + c_bar if problem.value is not None else None
        diagnostics["fallback_solver"] = "SCS"

    diagnostics["solve_time_seconds"] = time.time() - t0

    if w.value is None:
        raise RuntimeError(f"QP solver failed with status: {problem.status}")

    w_star = np.array(w.value, dtype=np.float64).flatten()
    cap = 1.0 / budget_k
    w_star = _project_capped_simplex(w_star, cap)

    if w_star.sum() < 1e-6:
        raise RuntimeError(
            f"QP solution collapsed to zero after projection "
            f"(psd_shift={diagnostics.get('psd_correction')})."
        )

    diagnostics["w_star_sum"] = float(w_star.sum())
    diagnostics["w_star_nnz"] = int(np.sum(w_star > 1e-8))
    diagnostics["w_star_max"] = float(w_star.max())
    diagnostics["w_star_min"] = float(w_star.min())

    return w_star, diagnostics


def three_way_split(
    n: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    total_requested = n_train + n_val + n_test
    if total_requested > n:
        raise ValueError(
            f"Requested split sizes ({n_train}+{n_val}+{n_test}={total_requested}) "
            f"exceed the number of available prompts ({n})"
        )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return (
        sorted(indices[:n_train].tolist()),
        sorted(indices[n_train:n_train + n_val].tolist()),
        sorted(indices[n_train + n_val:n_train + n_val + n_test].tolist()),
    )


def compute_averaged_kernels(
    prompts: list,
    train_indices: list[int],
    embedder,
    kernel: EmbeddingKernel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    T = prompts[train_indices[0]].num_steps
    A_V_sum = np.zeros((T, T), dtype=np.float64)
    A_U_sum = np.zeros((T, T), dtype=np.float64)
    C_sum = np.zeros((T, T), dtype=np.float64)
    n_valid = 0

    for count, idx in enumerate(train_indices):
        if count % 50 == 0:
            print(f"  [{count+1}/{len(train_indices)}] prompt={prompts[idx].prompt_id}")
        emb, N, _ = step_embeddings(prompts[idx], embedder)
        if emb.size == 0 or N < 2:
            continue
        A_V, A_U, C_mat = compute_kernel_matrices(emb, kernel)
        A_V_sum += A_V
        A_U_sum += A_U
        C_sum += C_mat
        n_valid += 1

    if n_valid == 0:
        raise RuntimeError("No valid training prompts found")

    return A_V_sum / n_valid, A_U_sum / n_valid, C_sum / n_valid, n_valid


def train(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Loading run from {run_dir}")
    run = load_generated_run(run_dir, step_view="x0")
    prompts = list(run.prompts)

    if not prompts:
        raise RuntimeError(f"No prompts found in run directory: {run_dir}")

    T = Counter(p.num_steps for p in prompts).most_common(1)[0][0]
    print(f"[train] Loaded {len(prompts)} prompts, most common step count T={T}")

    for p in prompts:
        if p.num_steps != T:
            print(f"[WARNING] Prompt {p.prompt_id} has {p.num_steps} steps (expected {T}), skipping")
    prompts = [p for p in prompts if p.num_steps == T]

    train_indices, val_indices, test_indices = three_way_split(
        len(prompts), args.n_train, args.n_val, args.n_test, args.split_seed,
    )
    print(f"[train] Split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test (seed={args.split_seed})")

    kernel = EmbeddingKernel(
        name=f"embedding-{args.kernel}",
        embedding_model=args.embedding_model,
        rbf_bandwidth=args.rbf_bandwidth,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        normalize_embeddings=True,
    )
    embedder = kernel.embedder

    t_optim_start = time.time()
    print(f"[train] Computing kernel matrices over {len(train_indices)} training prompts...")
    A_V_bar, A_U_bar, C_bar, n_valid_train = compute_averaged_kernels(
        prompts, train_indices, embedder, kernel,
    )
    kernel_time = time.time() - t_optim_start
    print(f"[train] Kernel matrices computed in {kernel_time:.2f}s ({n_valid_train} valid)")
    Delta_U_bar = compute_delta_U(C_bar, A_U_bar)

    A_bar = A_U_bar if args.estimator_type == "unbiased" else A_V_bar
    b_bar = A_bar[:, -1]
    c_bar = A_bar[-1, -1]

    print(f"\n[train] Matrix diagnostics (n_valid_train={n_valid_train}):")
    psd_checks = {}
    for name, mat in [("A_V_bar", A_V_bar), ("A_U_bar", A_U_bar), ("C_bar", C_bar), ("Delta_U_bar", Delta_U_bar)]:
        info = verify_psd(mat, name)
        psd_checks[name] = info
        status = "PSD" if info["is_psd"] else "NOT PSD"
        print(f"  {name}: {status} (min_eig={info['min_eigenvalue']:.6e}, max_eig={info['max_eigenvalue']:.6e})")

    print(f"\n[train] Solving QP (estimator={args.estimator_type}, lambda={args.lambda_reg}, K={args.budget_k})...")
    w_star, qp_diag = solve_qp(
        A_bar, b_bar, Delta_U_bar, c_bar,
        lambda_reg=args.lambda_reg,
        budget_k=args.budget_k,
    )

    qp_time = qp_diag["solve_time_seconds"]
    total_optim_time = kernel_time + qp_time

    print(f"  QP status: {qp_diag['status']}")
    print(f"  w* sum: {qp_diag['w_star_sum']:.6f}")
    print(f"  w* nnz (>1e-8): {qp_diag['w_star_nnz']}")
    print(f"  Objective value: {qp_diag['optimal_value']}")
    print(f"  QP solve time: {qp_time:.2f}s")
    print(f"[train] Total optimization time: {total_optim_time:.2f}s")

    np.savez(
        output_dir / "trained_weights.npz",
        w_star=w_star,
        A_V_bar=A_V_bar, A_U_bar=A_U_bar, C_bar=C_bar,
        Delta_U_bar=Delta_U_bar, b_bar=b_bar,
        train_indices=np.array(train_indices),
        val_indices=np.array(val_indices),
        test_indices=np.array(test_indices),
    )

    result = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "T": T,
        "n_prompts_total": len(prompts),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "n_valid_train": n_valid_train,
        "split_seed": args.split_seed,
        "estimator_type": args.estimator_type,
        "lambda_reg": args.lambda_reg,
        "budget_k": args.budget_k,
        "embedding_model": args.embedding_model,
        "kernel": args.kernel,
        "rbf_bandwidth": args.rbf_bandwidth,
        "w_star": w_star.tolist(),
        "timing": {
            "kernel_matrices_seconds": kernel_time,
            "qp_solve_seconds": qp_time,
            "total_optimization_seconds": total_optim_time,
        },
        "qp_diagnostics": qp_diag,
        "psd_checks": {k: {kk: vv for kk, vv in v.items() if kk != "eigenvalues"} for k, v in psd_checks.items()},
        "train_prompt_ids": [prompts[i].prompt_id for i in train_indices],
        "val_prompt_ids": [prompts[i].prompt_id for i in val_indices],
        "test_prompt_ids": [prompts[i].prompt_id for i in test_indices],
    }

    with (output_dir / "train_result.json").open("w") as f:
        json.dump(to_jsonable(result), f, indent=2, allow_nan=True)

    print(f"\n[train] Saved weights to {output_dir / 'trained_weights.npz'}")
    print(f"[train] Saved result to {output_dir / 'train_result.json'}")
    print("\n[train] Top-10 weights:")
    top_k_idx = np.argsort(w_star)[::-1][:10]
    for rank, idx in enumerate(top_k_idx):
        print(f"  rank={rank+1}  step={idx}  w={w_star[idx]:.6f}")

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal weights via capped simplex QP")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_train", type=int, required=True)
    parser.add_argument("--n_val", type=int, required=True)
    parser.add_argument("--n_test", type=int, required=True)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--estimator_type", choices=["unbiased", "convex_surrogate"], default="unbiased")
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--budget_k", type=int, default=8)
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "cosine"])
    parser.add_argument("--rbf_bandwidth", default="median")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--embedding_device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
