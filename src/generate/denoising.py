"""Pure diffusion denoising loop with per-model entry points.

Public API:
  generate(model, prompts, device, *, backend, ...)
  generate_llada(model, prompts, device, *, ...)
  generate_dream(model, prompts, device, *, ...)
  generate_nemotron(model, prompts, device, *, ...)

All return (answers: list[str], rich_traces: dict[str, Tensor])
with identical trace tensor shapes for downstream compatibility.
"""

from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def add_gumbel_noise(logits: torch.Tensor, temperature: float, *, _noise_buf: torch.Tensor | None = None) -> torch.Tensor:
    if temperature == 0:
        return logits
    noisy_logits = logits.float()
    if noisy_logits.data_ptr() == logits.data_ptr():
        noisy_logits = noisy_logits.clone()
    if _noise_buf is not None and _noise_buf.shape == noisy_logits.shape and _noise_buf.device == noisy_logits.device:
        noise = _noise_buf
    else:
        noise = torch.empty_like(noisy_logits, dtype=torch.float32)
    noise.uniform_()
    tiny = torch.finfo(noise.dtype).tiny
    noise.clamp_(min=tiny, max=1.0 - torch.finfo(noise.dtype).eps)
    noise.log_().neg_().log_().neg_()
    noisy_logits.add_(noise, alpha=float(temperature))
    return noisy_logits


def _selected_token_probability(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    selected_logits = torch.gather(logits, dim=-1, index=torch.unsqueeze(token_ids, -1)).squeeze(-1)
    normalizer = torch.logsumexp(logits, dim=-1)
    return torch.exp(selected_logits - normalizer)


def _mask_undecodable_token_logits(logits: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    invalid_ids = _invalid_token_ids_for_tokenizer(tokenizer, logits.shape[-1])
    if invalid_ids.numel() == 0:
        return logits
    cache = getattr(tokenizer, "_uq_invalid_device_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(tokenizer, "_uq_invalid_device_cache", cache)
        except Exception:
            cache = {}
    key = (int(logits.shape[-1]), str(logits.device))
    device_ids = cache.get(key)
    if device_ids is None:
        device_ids = invalid_ids.to(logits.device)
        cache[key] = device_ids
    logits.index_fill_(dim=-1, index=device_ids, value=-torch.inf)
    return logits


def _invalid_token_ids_for_tokenizer(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    if tokenizer is None or vocab_size is None:
        return torch.empty(0, dtype=torch.long)
    vocab_size = int(vocab_size)
    cache = getattr(tokenizer, "_uq_invalid_ids_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(tokenizer, "_uq_invalid_ids_cache", cache)
        except Exception:
            cache = {}
    if vocab_size in cache:
        return cache[vocab_size]

    valid_ids = set()
    has_source = False
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
            has_source = True
            valid_ids.update(int(tid) for tid in vocab.values() if tid is not None and 0 <= int(tid) < vocab_size)
        except Exception:
            valid_ids.clear()
            has_source = False

    if not has_source:
        decoder = getattr(tokenizer, "decoder", None)
        if isinstance(decoder, dict):
            has_source = True
            valid_ids.update(int(tid) for tid, tok in decoder.items() if tok is not None and 0 <= int(tid) < vocab_size)

    if not has_source or not valid_ids:
        invalid = torch.empty(0, dtype=torch.long)
    else:
        invalid = torch.tensor([tid for tid in range(vocab_size) if tid not in valid_ids], dtype=torch.long)
    cache[vocab_size] = invalid
    return invalid


def _non_special_token_counts(tokenizer: Any, token_ids: torch.Tensor, extra_special_ids: tuple = ()) -> torch.Tensor:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pid = getattr(tokenizer, "pad_token_id", None)
    if pid is not None:
        special_ids.add(int(pid))
    for tid in extra_special_ids or ():
        if tid is not None:
            special_ids.add(int(tid))
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    special_mask = torch.zeros_like(ids, dtype=torch.bool)
    for tid in special_ids:
        special_mask |= ids == int(tid)
    return (~special_mask).sum(dim=-1).detach().to(torch.int32).cpu()


def _merge_trace_batches(trace_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not trace_batches:
        return {}
    expected = set(trace_batches[0])
    for i, batch in enumerate(trace_batches[1:], start=1):
        if set(batch) != expected:
            raise ValueError(f"Trace batch {i} has inconsistent keys")
    return {key: torch.cat([b[key] for b in trace_batches], dim=0) for key in sorted(expected)}


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, dict[str, Any]] | None = None


def _get_backend_config(backend: str) -> dict[str, Any]:
    from src.generate._forward import llada_logits, dream_logits

    global _BACKENDS
    if _BACKENDS is None:
        _BACKENDS = {
            "llada": dict(get_logits=llada_logits),
            "dream": dict(get_logits=dream_logits),
        }
    cfg = _BACKENDS.get(backend)
    if cfg is None:
        raise ValueError(f"Unknown backend: {backend!r}. Valid: {sorted(_BACKENDS)}")
    return cfg


# ---------------------------------------------------------------------------
# Pure diffusion denoising loop + per-model entry points
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: Any,
    prompts: list[str],
    device: Any,
    *,
    backend: str = "llada",
    batch_size: int = 8,
    tokenizer: Any = None,
    steps: int = 128,
    gen_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    eos_token_ids: list[int] | tuple[int, ...] | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    save_trajectory: bool = True,
    k_topk_logits: int = 64,
    save_token_logprobs: bool = True,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    if backend == "nemotron":
        return generate_nemotron(
            model, prompts, device,
            batch_size=batch_size, tokenizer=tokenizer, steps=steps,
            gen_length=gen_length, temperature=temperature,
            remasking=remasking, mask_id=mask_id,
            eos_token_ids=eos_token_ids, logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
            save_trajectory=save_trajectory, k_topk_logits=k_topk_logits,
            save_token_logprobs=save_token_logprobs,
        )

    cfg = _get_backend_config(backend)
    get_logits = cfg["get_logits"]

    if "setup_fn" in cfg:
        cfg["setup_fn"](model)

    generated_answers: list[str] = []
    trace_batches: list[dict[str, torch.Tensor]] = []
    save_topk = save_trajectory and k_topk_logits is not None
    noise_buf = None

    for batch_start in tqdm(
        range(0, len(prompts), batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc="Generating batches",
    ):
        prompt_batch = prompts[batch_start:batch_start + batch_size]
        encoded = tokenizer(prompt_batch, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        x = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        full_attention_mask = torch.cat(
            [attention_mask, torch.ones((input_ids.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)],
            dim=-1,
        )
        response_slice = slice(input_ids.shape[1], input_ids.shape[1] + gen_length)

        if save_trajectory:
            intermediate_x0_pred = []
            response_token_steps = []
            mask_status_steps = []
            newly_unmasked_steps = []
            remasked_steps = []
            x_gen_count_steps = []
            x0_gen_count_steps = []
            if save_token_logprobs:
                x0_logprobs_steps = []
            previous_mask = torch.ones((input_ids.shape[0], gen_length), dtype=torch.bool, device=model.device)
        if save_topk:
            topk_logits_steps = []
            topk_ids_steps = []

        num_unmasked_per_step = max(1, (gen_length + steps - 1) // steps)

        for step_i in range(steps):
            response_state = x[:, response_slice]
            response_mask_index = response_state == mask_id

            full_logits = get_logits(model, x, full_attention_mask)
            response_logits = full_logits[:, response_slice, :].contiguous()
            del full_logits

            if logits_eos_inf and eos_token_ids:
                for _eos_id in eos_token_ids:
                    if 0 <= _eos_id < response_logits.shape[-1]:
                        response_logits[:, :, _eos_id] = -torch.inf
            response_logits = response_logits.float()
            response_logits = _mask_undecodable_token_logits(response_logits, tokenizer)

            if save_topk and step_i == steps - 1:
                tk_logits, tk_ids = torch.topk(response_logits, k=min(int(k_topk_logits), response_logits.shape[-1]), dim=-1)
                topk_logits_steps.append(tk_logits.detach().to(torch.float16).cpu())
                topk_ids_steps.append(tk_ids.detach().to(torch.int32).cpu())
                del tk_logits, tk_ids

            if temperature > 0 and (noise_buf is None or noise_buf.shape != response_logits.shape):
                noise_buf = torch.empty(response_logits.shape, dtype=torch.float32, device=response_logits.device)
            logits_noisy = add_gumbel_noise(response_logits, temperature=temperature, _noise_buf=noise_buf)
            x0_response = torch.argmax(logits_noisy, dim=-1)
            del logits_noisy

            if save_trajectory:
                x0_greedy = torch.argmax(response_logits, dim=-1)
                x0_obs_ids = torch.where(response_mask_index, x0_greedy, response_state)
                if save_token_logprobs:
                    sel = torch.gather(response_logits, dim=-1, index=x0_obs_ids.unsqueeze(-1)).squeeze(-1)
                    norm = torch.logsumexp(response_logits, dim=-1)
                    x0_logprobs_steps.append((sel - norm).detach().to(torch.float32).cpu())
                intermediate_x0_pred.append(x0_obs_ids.detach().to(torch.int32).cpu())
                x0_gen_count_steps.append(_non_special_token_counts(tokenizer, x0_obs_ids, extra_special_ids=(mask_id,)))
                del x0_obs_ids, x0_greedy

            if remasking == "low_confidence":
                logits_conf = response_logits
                if confidence_eos_eot_inf and eos_token_ids:
                    logits_conf = response_logits.clone()
                    for _eos_id in eos_token_ids:
                        if 0 <= _eos_id < logits_conf.shape[-1]:
                            logits_conf[:, :, _eos_id] = -torch.inf
                x0_p = _selected_token_probability(logits_conf, x0_response)
                if logits_conf is not response_logits:
                    del logits_conf
            elif remasking == "random":
                x0_p = torch.rand(x0_response.shape, device=x0_response.device)
            else:
                raise ValueError(f"Unsupported remasking: {remasking}")

            x0_response = torch.where(response_mask_index, x0_response, response_state)
            confidence = torch.where(response_mask_index, x0_p, -torch.inf)

            transfer = torch.zeros_like(response_mask_index, dtype=torch.bool, device=x.device)
            _, sel_idx = torch.topk(confidence, k=min(num_unmasked_per_step, confidence.shape[1]), largest=True, dim=-1)
            transfer.scatter_(dim=-1, index=sel_idx, value=True)
            transfer &= response_mask_index
            response_state[transfer] = x0_response[transfer]

            if save_trajectory:
                cur_mask = response_state == mask_id
                response_token_steps.append(response_state.detach().to(torch.int32).cpu())
                mask_status_steps.append(cur_mask.detach().cpu())
                newly_unmasked_steps.append((previous_mask & ~cur_mask).detach().cpu())
                remasked_steps.append((~previous_mask & cur_mask).detach().cpu())
                x_gen_count_steps.append(_non_special_token_counts(tokenizer, response_state, extra_special_ids=(mask_id,)))
                previous_mask = cur_mask.clone()

            del response_logits, x0_response, x0_p, confidence, transfer

        response_ids = x[:, response_slice]
        generated_answers.extend(
            ans.strip() for ans in tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        )

        if save_trajectory:
            mask_status = torch.stack(mask_status_steps, dim=1).to(torch.bool)
            batch_trace = {
                "response_token_ids": torch.stack(response_token_steps, dim=1).to(torch.int32),
                "x0_pred_token_ids": torch.stack(intermediate_x0_pred, dim=1).to(torch.int32),
                "mask_status": mask_status,
                "unmasked_status": ~mask_status,
                "newly_unmasked": torch.stack(newly_unmasked_steps, dim=1).to(torch.bool),
                "remasked": torch.stack(remasked_steps, dim=1).to(torch.bool),
                "x_generated_token_counts": torch.stack(x_gen_count_steps, dim=1).to(torch.int32),
                "x0_generated_token_counts": torch.stack(x0_gen_count_steps, dim=1).to(torch.int32),
            }
            if save_token_logprobs and x0_logprobs_steps:
                batch_trace["x0_token_logprobs"] = torch.stack(x0_logprobs_steps, dim=1).to(torch.float32)
            if save_topk:
                batch_trace["topk_logits"] = torch.stack(topk_logits_steps, dim=1).to(torch.float16)
                batch_trace["topk_token_ids"] = torch.stack(topk_ids_steps, dim=1).to(torch.int32)
            trace_batches.append(batch_trace)

        del encoded, input_ids, attention_mask, full_attention_mask, x
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    rich_traces = _merge_trace_batches(trace_batches) if save_trajectory else {}
    return generated_answers, rich_traces


def generate_llada(model, prompts, device, **kwargs):
    return generate(model, prompts, device, backend="llada", **kwargs)


def generate_dream(model, prompts, device, **kwargs):
    return generate(model, prompts, device, backend="dream", **kwargs)


@torch.no_grad()
def generate_nemotron(
    model: Any,
    prompts: list[str],
    device: Any,
    *,
    batch_size: int = 8,
    tokenizer: Any = None,
    steps: int = 128,
    gen_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    eos_token_ids: list[int] | tuple[int, ...] | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    save_trajectory: bool = True,
    k_topk_logits: int = 64,
    save_token_logprobs: bool = True,
    backend: str = "nemotron",
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Nemotron-native diffusion generation with KV-cached prompt prefill.

    Pure diffusion (single block): causal prefill encodes the prompt into a KV
    cache, then denoising steps only forward the gen_length block.  This allows
    batching (padding is handled during causal prefill where attention_mask is
    respected) and reduces per-step cost from O(prompt+gen) to O(gen).
    """
    from src.generate._forward import nemotron_prefill, nemotron_block_logits

    generated_answers: list[str] = []
    trace_batches: list[dict[str, torch.Tensor]] = []
    save_topk = save_trajectory and k_topk_logits is not None
    noise_buf = None

    for batch_start in tqdm(
        range(0, len(prompts), batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc="Generating batches",
    ):
        prompt_batch = prompts[batch_start:batch_start + batch_size]
        encoded = tokenizer(prompt_batch, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        B = input_ids.shape[0]

        # -- Phase 1: causal prefill (bidirectional OFF) -----------------------
        # attention_mask is respected during causal attention, so padding is safe.
        past_key_values, prefill_logits = nemotron_prefill(model, input_ids, attention_mask)

        # -- Phase 2: build mask block -----------------------------------------
        block = torch.full((B, gen_length), mask_id, dtype=torch.long, device=device)

        if save_trajectory:
            intermediate_x0_pred = []
            response_token_steps = []
            mask_status_steps = []
            newly_unmasked_steps = []
            remasked_steps = []
            x_gen_count_steps = []
            x0_gen_count_steps = []
            if save_token_logprobs:
                x0_logprobs_steps = []
            previous_mask = torch.ones((B, gen_length), dtype=torch.bool, device=device)
        if save_topk:
            topk_logits_steps = []
            topk_ids_steps = []

        num_unmasked_per_step = max(1, (gen_length + steps - 1) // steps)

        # -- Phase 3: denoising loop (bidirectional ON via prefill toggle) -----
        for step_i in range(steps):
            mask_index = block == mask_id

            # Forward only the block against the cached prompt KV
            response_logits = nemotron_block_logits(model, block, past_key_values)

            if logits_eos_inf and eos_token_ids:
                for _eos_id in eos_token_ids:
                    if 0 <= _eos_id < response_logits.shape[-1]:
                        response_logits[:, :, _eos_id] = -torch.inf
            response_logits = response_logits.float()
            response_logits = _mask_undecodable_token_logits(response_logits, tokenizer)

            if save_topk and step_i == steps - 1:
                tk_logits, tk_ids = torch.topk(response_logits, k=min(int(k_topk_logits), response_logits.shape[-1]), dim=-1)
                topk_logits_steps.append(tk_logits.detach().to(torch.float16).cpu())
                topk_ids_steps.append(tk_ids.detach().to(torch.int32).cpu())
                del tk_logits, tk_ids

            if temperature > 0 and (noise_buf is None or noise_buf.shape != response_logits.shape):
                noise_buf = torch.empty(response_logits.shape, dtype=torch.float32, device=response_logits.device)
            logits_noisy = add_gumbel_noise(response_logits, temperature=temperature, _noise_buf=noise_buf)
            x0_sampled = torch.argmax(logits_noisy, dim=-1)
            del logits_noisy

            # -- x0 prediction trace (always greedy argmax, independent of sampling) --
            if save_trajectory:
                x0_greedy = torch.argmax(response_logits, dim=-1)
                x0_obs_ids = torch.where(mask_index, x0_greedy, block)
                if save_token_logprobs:
                    sel = torch.gather(response_logits, dim=-1, index=x0_obs_ids.unsqueeze(-1)).squeeze(-1)
                    norm = torch.logsumexp(response_logits, dim=-1)
                    x0_logprobs_steps.append((sel - norm).detach().to(torch.float32).cpu())
                intermediate_x0_pred.append(x0_obs_ids.detach().to(torch.int32).cpu())
                x0_gen_count_steps.append(_non_special_token_counts(tokenizer, x0_obs_ids, extra_special_ids=(mask_id,)))
                del x0_obs_ids, x0_greedy

            # -- Confidence-based token selection ----------------------------------
            if remasking == "low_confidence":
                logits_conf = response_logits
                if confidence_eos_eot_inf and eos_token_ids:
                    logits_conf = response_logits.clone()
                    for _eos_id in eos_token_ids:
                        if 0 <= _eos_id < logits_conf.shape[-1]:
                            logits_conf[:, :, _eos_id] = -torch.inf
                x0_p = _selected_token_probability(logits_conf, x0_sampled)
                if logits_conf is not response_logits:
                    del logits_conf
            elif remasking == "random":
                x0_p = torch.rand(x0_sampled.shape, device=x0_sampled.device)
            else:
                raise ValueError(f"Unsupported remasking: {remasking}")

            x0_sampled = torch.where(mask_index, x0_sampled, block)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer = torch.zeros_like(mask_index, dtype=torch.bool, device=device)
            _, sel_idx = torch.topk(confidence, k=min(num_unmasked_per_step, confidence.shape[1]), largest=True, dim=-1)
            transfer.scatter_(dim=-1, index=sel_idx, value=True)
            transfer &= mask_index
            block[transfer] = x0_sampled[transfer]

            if save_trajectory:
                cur_mask = block == mask_id
                response_token_steps.append(block.detach().to(torch.int32).cpu())
                mask_status_steps.append(cur_mask.detach().cpu())
                newly_unmasked_steps.append((previous_mask & ~cur_mask).detach().cpu())
                remasked_steps.append((~previous_mask & cur_mask).detach().cpu())
                x_gen_count_steps.append(_non_special_token_counts(tokenizer, block, extra_special_ids=(mask_id,)))
                previous_mask = cur_mask.clone()

            del response_logits, x0_sampled, x0_p, confidence, transfer

        # -- Decode final answers ----------------------------------------------
        generated_answers.extend(
            ans.strip() for ans in tokenizer.batch_decode(block, skip_special_tokens=True)
        )

        if save_trajectory:
            mask_status = torch.stack(mask_status_steps, dim=1).to(torch.bool)
            batch_trace = {
                "response_token_ids": torch.stack(response_token_steps, dim=1).to(torch.int32),
                "x0_pred_token_ids": torch.stack(intermediate_x0_pred, dim=1).to(torch.int32),
                "mask_status": mask_status,
                "unmasked_status": ~mask_status,
                "newly_unmasked": torch.stack(newly_unmasked_steps, dim=1).to(torch.bool),
                "remasked": torch.stack(remasked_steps, dim=1).to(torch.bool),
                "x_generated_token_counts": torch.stack(x_gen_count_steps, dim=1).to(torch.int32),
                "x0_generated_token_counts": torch.stack(x0_gen_count_steps, dim=1).to(torch.int32),
            }
            if save_token_logprobs and x0_logprobs_steps:
                batch_trace["x0_token_logprobs"] = torch.stack(x0_logprobs_steps, dim=1).to(torch.float32)
            if save_topk:
                batch_trace["topk_logits"] = torch.stack(topk_logits_steps, dim=1).to(torch.float16)
                batch_trace["topk_token_ids"] = torch.stack(topk_ids_steps, dim=1).to(torch.int32)
            trace_batches.append(batch_trace)

        del encoded, input_ids, attention_mask, past_key_values, block
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    rich_traces = _merge_trace_batches(trace_batches) if save_trajectory else {}
    return generated_answers, rich_traces
