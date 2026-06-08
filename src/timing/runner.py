"""Per-prompt orchestrator: time each UQ feature and aggregate over prompts.

Cost attribution: each feature row reports the generation(s) it requires plus
its own NLI + math cost. The baseline is token-based UQ (whose cost ≈ greedy
generation). Overhead for Pareto plotting = time_total - baseline.

Generation is measured three ways per prompt (independently):
  1. Greedy (T=0, batch=1) — one deterministic trajectory
  2. Single stochastic (T>0, batch=1) — one stochastic sample
  3. Batched stochastic (T>0, batch=N) — N samples in parallel

Within a prompt, NLI pairs and stochastic samples are fully batched.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.features.token import special_ids_from_config, visible_positions
from src.timing import generate as gen_mod
from src.timing import math_ops
from src.timing.nli import all_pairs_entail_matrix


SAMPLING_NLI_FEATURES = ("se-marginal", "se-conditional", "ecc", "eigval", "kle-heat", "kle-matern")

NLI_FEATURE_TIMERS = {
    "se-marginal": math_ops.time_se_marginal,
    "se-conditional": math_ops.time_se_conditional,
    "ecc": math_ops.time_ecc,
    "eigval": math_ops.time_eigval,
    "kle-heat": math_ops.time_kle_heat,
    "kle-matern": math_ops.time_kle_matern,
}


@dataclass
class PromptRecord:
    prompt_idx: int
    feature: str
    family: str
    n_extra_gen: int
    n_nli_pairs: int
    time_gen: float
    time_nli: float
    time_math: float

    @property
    def time_total(self) -> float:
        return self.time_gen + self.time_nli + self.time_math


@dataclass
class GenTimings:
    """Three independent generation timing measurements for one prompt."""
    greedy: float
    single_stochastic: float
    batched_stochastic: float
    n_samples: int


@dataclass
class RunDiagnostics:
    n_prompts: int
    n_warmup: int
    n_response_samples: int
    steps: int
    gen_length: int
    budget_k: int | None
    selected_step_indices: list[int] | None
    nli_batch_size: int
    greedy_gen_time_mean: float = 0.0
    greedy_gen_time_std: float = 0.0
    single_stochastic_gen_time_mean: float = 0.0
    single_stochastic_gen_time_std: float = 0.0
    batched_stochastic_gen_time_mean: float = 0.0
    batched_stochastic_gen_time_std: float = 0.0
    per_prompt_raw: list[PromptRecord] = field(default_factory=list)


def load_qp_weights(selection_dir: str | Path | None, *, budget_k: int | None) -> list[int] | None:
    """Load top-K step indices from a saved QP trained_weights.npz. None if unavailable."""
    if selection_dir is None or budget_k is None:
        return None
    path = Path(selection_dir) / "trained_weights.npz"
    if not path.exists():
        return None
    data = np.load(path)
    if "w_star" not in data.files:
        return None
    w_star = np.asarray(data["w_star"], dtype=np.float64)
    active = w_star[:-1]
    k = min(int(budget_k), active.size)
    return sorted(int(i) for i in np.argsort(active)[-k:])


def decode_x0_steps(x0_pred_token_ids: Any, tokenizer: Any, sample_idx: int = 0) -> list[str]:
    """Decode every step's x0 prediction into a text string."""
    arr = np.asarray(x0_pred_token_ids)
    if arr.ndim != 3 or arr.shape[0] <= sample_idx:
        return []
    t_steps = arr.shape[1]
    steps = []
    for t in range(t_steps):
        token_ids = arr[sample_idx, t].tolist()
        steps.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())
    return steps


def _extract_token_arrays(traces: dict, sample_idx: int = 0) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    lp = traces.get("x0_token_logprobs")
    tk = traces.get("topk_logits")
    tids = traces.get("response_token_ids")
    final_lp = np.asarray(lp)[sample_idx, -1] if lp is not None else None
    final_tk = np.asarray(tk)[sample_idx, -1] if tk is not None else None
    final_tids = np.asarray(tids)[sample_idx, -1] if tids is not None else None
    return final_lp, final_tk, final_tids


def _sampled_logprobs_and_ids(traces: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    lp = traces.get("x0_token_logprobs")
    tids = traces.get("response_token_ids")
    if lp is None or tids is None:
        return [], []
    lp_arr = np.asarray(lp)
    tid_arr = np.asarray(tids)
    n = lp_arr.shape[0]
    return [lp_arr[i, -1] for i in range(n)], [tid_arr[i, -1] for i in range(n)]


def profile_one_prompt(
    prompt_idx: int,
    prompt: str,
    *,
    model: Any,
    tokenizer: Any,
    nli_model: Any,
    config: dict,
    eos_token_ids: list[int],
    device: Any,
    selected_steps: list[int] | None,
    nli_batch_size: int,
    full_trajectory: bool = False,
) -> tuple[list[PromptRecord], GenTimings]:
    """Profile every feature for one prompt.

    Returns (records, gen_timings) where gen_timings holds the three
    independent generation measurements.
    """
    n_response_samples = int(config.get("num_response_samples", 20))

    # --- Three independent generation measurements ---

    # 1. Greedy trajectory (T=0, batch=1)
    greedy_gen_time, greedy_answer, greedy_traces = gen_mod.time_greedy_one(
        model, prompt, config=config, tokenizer=tokenizer, eos_token_ids=eos_token_ids, device=device,
    )

    # 2. Single stochastic sample (T>0, batch=1)
    single_stoch_time, _, _ = gen_mod.time_single_stochastic(
        model, prompt, config=config, tokenizer=tokenizer, eos_token_ids=eos_token_ids, device=device,
    )

    # 3. N stochastic samples batched (T>0, batch=N)
    batched_stoch_time, sampled_answers, sampled_traces = gen_mod.time_batched_stochastic(
        model, prompt, n_samples=n_response_samples, config=config,
        tokenizer=tokenizer, eos_token_ids=eos_token_ids, device=device,
    )

    gen_timings = GenTimings(
        greedy=greedy_gen_time,
        single_stochastic=single_stoch_time,
        batched_stochastic=batched_stoch_time,
        n_samples=n_response_samples,
    )

    # --- Extract arrays for feature computation ---

    special, eos = special_ids_from_config(config)
    final_lp, final_tk, final_tids = _extract_token_arrays(greedy_traces, sample_idx=0)
    vis = visible_positions(final_tids, special_ids=special, eos_ids=eos) if final_tids is not None else []

    sampled_lps, sampled_tids = _sampled_logprobs_and_ids(sampled_traces) if sampled_traces else ([], [])

    iid_texts = list(sampled_answers)
    step_strings = decode_x0_steps(greedy_traces.get("x0_pred_token_ids"), tokenizer, sample_idx=0)
    selected_step_strings = [step_strings[i] for i in selected_steps] if selected_steps else []

    # --- NLI matrices (batched within prompt) ---

    iid_nli_time, iid_sym, iid_pairs = all_pairs_entail_matrix(
        nli_model, iid_texts, nli_batch_size=nli_batch_size, device=device,
    )

    if full_trajectory:
        full_nli_time, full_sym, full_pairs = all_pairs_entail_matrix(
            nli_model, step_strings, nli_batch_size=nli_batch_size, device=device,
        )
    else:
        full_nli_time, full_sym, full_pairs = 0.0, None, 0

    if selected_step_strings:
        sel_nli_time, sel_sym, sel_pairs = all_pairs_entail_matrix(
            nli_model, selected_step_strings, nli_batch_size=nli_batch_size, device=device,
        )
    else:
        sel_nli_time, sel_sym, sel_pairs = 0.0, None, 0

    # --- Build per-feature records ---

    records: list[PromptRecord] = []

    # Token features: cost = greedy generation + math
    if final_lp is not None:
        t_msp, _ = math_ops.time_msp(final_lp, vis=vis)
        t_ppl, _ = math_ops.time_perplexity(final_lp, vis=vis)
        records.append(PromptRecord(prompt_idx, "msp", "token", 0, 0, greedy_gen_time, 0.0, t_msp))
        records.append(PromptRecord(prompt_idx, "perplexity", "token", 0, 0, greedy_gen_time, 0.0, t_ppl))
    if final_tk is not None:
        t_mte, _ = math_ops.time_mte(final_tk, vis=vis)
        records.append(PromptRecord(prompt_idx, "mte", "token", 0, 0, greedy_gen_time, 0.0, t_mte))

    # mcnse: cost = batched stochastic generation + math (no NLI)
    if sampled_lps:
        t_mcnse, _ = math_ops.time_mcnse(sampled_lps, sampled_tids, special_ids=special, eos_ids=eos)
        records.append(PromptRecord(prompt_idx, "mcnse", "iid", n_response_samples, 0, batched_stoch_time, 0.0, t_mcnse))

    # iid NLI features: cost = batched stochastic generation + NLI + math
    for feat in SAMPLING_NLI_FEATURES:
        t_math, _ = NLI_FEATURE_TIMERS[feat](iid_sym)
        records.append(PromptRecord(prompt_idx, feat, "iid", n_response_samples, iid_pairs, batched_stoch_time, iid_nli_time, t_math))

    # Full-trajectory features: cost = greedy generation + NLI(all steps) + math
    if full_sym is not None:
        for feat in SAMPLING_NLI_FEATURES:
            t_math, _ = NLI_FEATURE_TIMERS[feat](full_sym)
            records.append(PromptRecord(prompt_idx, f"full-{feat}", "full", 0, full_pairs, greedy_gen_time, full_nli_time, t_math))

    # Selected-step features: cost = greedy generation + NLI(K steps) + math
    if sel_sym is not None:
        for feat in SAMPLING_NLI_FEATURES:
            t_math, _ = NLI_FEATURE_TIMERS[feat](sel_sym)
            records.append(PromptRecord(prompt_idx, f"selected-{feat}", "selected", 0, sel_pairs, greedy_gen_time, sel_nli_time, t_math))

    return records, gen_timings


def aggregate(records: list[PromptRecord]) -> dict[tuple[str, str], dict[str, float | int]]:
    """Group raw records by (family, feature), compute mean/std/totals."""
    grouped: dict[tuple[str, str], list[PromptRecord]] = {}
    for r in records:
        grouped.setdefault((r.family, r.feature), []).append(r)

    summary: dict[tuple[str, str], dict[str, float | int]] = {}
    for key, rs in grouped.items():
        gen = np.array([r.time_gen for r in rs], dtype=np.float64)
        nli = np.array([r.time_nli for r in rs], dtype=np.float64)
        math_t = np.array([r.time_math for r in rs], dtype=np.float64)
        total = np.array([r.time_total for r in rs], dtype=np.float64)
        ddof = 1 if len(rs) > 1 else 0
        summary[key] = {
            "n_prompts": len(rs),
            "n_extra_gen": rs[0].n_extra_gen,
            "n_nli_pairs": rs[0].n_nli_pairs,
            "time_gen_mean": float(gen.mean()),
            "time_gen_std": float(gen.std(ddof=ddof)),
            "time_nli_mean": float(nli.mean()),
            "time_nli_std": float(nli.std(ddof=ddof)),
            "time_math_mean": float(math_t.mean()),
            "time_math_std": float(math_t.std(ddof=ddof)),
            "time_total_mean": float(total.mean()),
            "time_total_std": float(total.std(ddof=ddof)),
        }
    return summary


def sample_prompts(config: dict, tokenizer: Any, *, num_prompts: int, hf_token: str | None, seed: int) -> list[str]:
    """Pull a fresh random sample of `num_prompts` from the HF dataset."""
    from src.generate.dataset_inputs import prepare_dataset_inputs

    cfg = dict(config)
    cfg["num_questions"] = max(num_prompts * 4, num_prompts + 1)
    _, _, prompts = prepare_dataset_inputs(cfg, tokenizer, hf_token)
    rng = random.Random(seed)
    indices = list(range(len(prompts)))
    rng.shuffle(indices)
    chosen = indices[:num_prompts]
    return [prompts[i] for i in chosen]
