"""Generation timing wrappers for latency profiling.

Three independent generation measurements per prompt:
  1. Greedy (T=0, batch=1) — single deterministic decode
  2. Single stochastic (T>0, batch=1) — one stochastic sample
  3. Batched stochastic (T>0, batch=N) — N samples in one batched call

Within a prompt, stochastic samples are batched together (batch_size=N).
"""

from __future__ import annotations

from typing import Any

from src.config import resolve_remasking
from src.generate import generate
from src.timing.clock import Timer


def _base_gen_kwargs(config: dict, *, tokenizer: Any, eos_token_ids: list[int], device: Any) -> dict:
    remasking_full = resolve_remasking(config["remasking"])
    return dict(
        backend=config.get("model_backend", "llada"),
        device=device,
        tokenizer=tokenizer,
        steps=int(config["steps"]),
        gen_length=int(config["gen_length"]),
        remasking=remasking_full,
        mask_id=int(config["mask_id"]),
        eos_token_ids=eos_token_ids,
        logits_eos_inf=bool(config.get("logits_eos_inf", False)),
        confidence_eos_eot_inf=bool(config.get("confidence_eos_eot_inf", False)),
        save_trajectory=True,
    )


def time_greedy_one(
    model: Any,
    prompt: str,
    *,
    config: dict,
    tokenizer: Any,
    eos_token_ids: list[int],
    device: Any,
) -> tuple[float, str, dict]:
    """Time a single greedy generation (T=0, batch=1). Returns (elapsed, answer, rich_traces)."""
    kwargs = _base_gen_kwargs(config, tokenizer=tokenizer, eos_token_ids=eos_token_ids, device=device)
    kwargs["batch_size"] = 1
    kwargs["k_topk_logits"] = int(config.get("topk_trace_k", 64))
    with Timer(device) as t:
        answers, traces = generate(model=model, prompts=[prompt], temperature=0.0, **kwargs)
    return float(t.elapsed), answers[0], traces


def time_single_stochastic(
    model: Any,
    prompt: str,
    *,
    config: dict,
    tokenizer: Any,
    eos_token_ids: list[int],
    device: Any,
) -> tuple[float, str, dict]:
    """Time a single stochastic generation (T>0, batch=1). Returns (elapsed, answer, traces)."""
    kwargs = _base_gen_kwargs(config, tokenizer=tokenizer, eos_token_ids=eos_token_ids, device=device)
    kwargs["batch_size"] = 1
    kwargs["k_topk_logits"] = None
    temperature = float(config.get("temperature", 1.0))
    with Timer(device) as t:
        answers, traces = generate(model=model, prompts=[prompt], temperature=temperature, **kwargs)
    return float(t.elapsed), answers[0], traces


def time_batched_stochastic(
    model: Any,
    prompt: str,
    *,
    n_samples: int,
    config: dict,
    tokenizer: Any,
    eos_token_ids: list[int],
    device: Any,
) -> tuple[float, list[str], dict]:
    """Time N stochastic samples generated in a single batched call.

    batch_size=N so the GPU processes all samples in parallel.
    Returns (elapsed, answers, traces).
    """
    if n_samples <= 0:
        return 0.0, [], {}
    kwargs = _base_gen_kwargs(config, tokenizer=tokenizer, eos_token_ids=eos_token_ids, device=device)
    kwargs["batch_size"] = n_samples
    kwargs["k_topk_logits"] = None
    temperature = float(config.get("temperature", 1.0))
    prompts = [prompt] * n_samples
    with Timer(device) as t:
        answers, traces = generate(model=model, prompts=prompts, temperature=temperature, **kwargs)
    return float(t.elapsed), answers, traces
