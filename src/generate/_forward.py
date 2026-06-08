"""Per-model forward functions for diffusion denoising.

Each function has the signature (model, state, attention_mask) -> logits
where logits[i] predicts the token at position i (LLaDA convention).
"""

from __future__ import annotations

from typing import Any

import torch


def llada_logits(model: Any, state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return model(state, attention_mask=attention_mask).logits


def dream_logits(model: Any, state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    attention, tok_idx = _dream_attention_for_full_mask(attention_mask, state.device)
    logits = _forward_dream_state(model, state, attention, tok_idx).logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def _set_nemotron_diffusion_lm(model: Any, enabled: bool) -> None:
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = enabled


def nemotron_prefill(model: Any, prompt_ids: torch.Tensor, attention_mask: torch.Tensor):
    _set_nemotron_diffusion_lm(model, False)
    output = model(prompt_ids, attention_mask=attention_mask, use_cache=True, use_causal_mask=True)
    _set_nemotron_diffusion_lm(model, True)
    return output.past_key_values, output.logits


def nemotron_block_logits(model: Any, block: torch.Tensor, past_key_values: Any) -> torch.Tensor:
    return model(block, past_key_values=past_key_values, use_cache=False).logits


def _dream_attention_for_full_mask(full_attention_mask: torch.Tensor, device: Any):
    attention_mask = full_attention_mask.to(device)
    if torch.any(attention_mask == 0):
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        attention = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        return attention, tok_idx
    return "full", None


def _forward_dream_state(model: Any, state: torch.Tensor, attention: Any, tok_idx: Any):
    attempts = [
        lambda: model(state, attention, tok_idx),
        lambda: model(state, attention_mask=attention, position_ids=tok_idx),
    ]
    if not isinstance(attention, str):
        attempts.append(lambda: model(state, attention_mask=attention))
    attempts.append(lambda: model(state))
    last_exc = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_exc = exc
    raise last_exc
