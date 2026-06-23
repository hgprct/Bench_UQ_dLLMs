"""Dedicated Dream (Dream-org/Dream-v0-Instruct-7B) diffusion inference.

Dream is an open 7B diffusion LLM (Qwen2 backbone). We re-use its *own* generation
entry point ``DreamModel.diffusion_generate`` (shipped in the repo's
``generation_utils.py`` and loaded via ``trust_remote_code``) rather than
re-implementing a denoising loop -- staying as close to the plain HF interface as
the rest of the bench.

Dream's denoising is *full-sequence* (there is no block-diffusion knob): at each
of ``steps`` timesteps it unmasks the tokens selected by the ``alg`` confidence
rule. The supported algorithms (see ``generation_utils.py``) are::

    'origin'        schedule-based unmasking, no confidence   ~ "random"
    'maskgit_plus'  unmask the highest-probability tokens
    'topk_margin'   unmask the highest top-2 margin tokens
    'entropy'       unmask the lowest-entropy (most confident) ~ "low_confidence"

We map the bench's ``remasking`` knob onto these (low_confidence -> entropy,
random -> origin); pass an explicit ``alg`` to override.

On top of generation we add one post-hoc forward pass over the completed sequence
to read top-k log-probs, so this returns the **same** ``(answers, topk_data)``
contract as ``src.generate.denoising.generate`` and plugs into the rest of the
pipeline unchanged. Dream's ``forward`` returns a ``MaskedLMOutput`` whose
``.logits`` span the full vocabulary, exactly like LLaDA.

Public API:
  generate(model, prompts, device, *, tokenizer, ...) -> (answers, topk_data)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Dream's diffusion mask token id (config.json: mask_token_id = 151666).
DREAM_MASK_ID: int = 151666

# Bench remasking name -> Dream alg. low_confidence -> the most-confident-first
# 'entropy' rule (recommended for the instruct model); random -> 'origin'.
_REMASK_TO_ALG: dict[str, str] = {
    "low_confidence": "entropy",
    "random": "origin",
}


@torch.no_grad()
def generate(
    model: Any,
    prompts: list[str],
    device: Any,
    *,
    tokenizer: Any,
    steps: int = 128,
    max_gen_length: int = 128,
    block_size: int | None = None,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = DREAM_MASK_ID,
    eos_token_ids: list[int] | tuple[int, ...] | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    batch_size: int = 8,
    top_k: int = 64,
    # Dream-native knobs (override the remasking-derived defaults if set).
    alg: str | None = None,
    alg_temp: float | None = 0.0,
    dream_top_p: float | None = None,
    dream_top_k: int | None = None,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Generate answers for text prompts using Dream's diffusion_generate.

    ``block_size``/``cfg_scale``/``logits_eos_inf``/``confidence_eos_eot_inf`` are
    accepted for interface parity with ``denoising.generate`` but are inert here:
    Dream is full-sequence (no blocking) and has no CFG / EOS-logit knobs.
    Returns the same ``(answers, topk_data)`` contract.
    """
    resolved_alg = alg or _REMASK_TO_ALG.get(remasking, "entropy")
    is_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    all_answers: list[str] = []
    topk_logprobs_batches: list[torch.Tensor] = []
    topk_ids_batches: list[torch.Tensor] = []

    for batch_start in tqdm(
        range(0, len(prompts), batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc="Generating (dream)",
    ):
        prompt_batch = prompts[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            prompt_batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]
        del encoded

        out = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_gen_length,
            steps=steps,
            temperature=temperature,
            top_p=dream_top_p,
            top_k=dream_top_k,
            alg=resolved_alg,
            alg_temp=alg_temp,
            mask_token_id=mask_id,
            output_history=False,
            return_dict_in_generate=True,
        )
        x = out.sequences
        del input_ids, out
        response_ids = x[:, prompt_len:]

        # Post-hoc top-k log-probs over the answer region. The generated region
        # was padded/unmasked by diffusion_generate, so the prompt-side attention
        # mask is extended with ones over the answer span.
        full_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (x.shape[0], x.shape[1] - prompt_len),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )
        # DreamSdpaAttention forwards the mask straight to
        # scaled_dot_product_attention, which (torch>=2.7) needs a bool/float 4D
        # mask, not the 2D long padding mask. Build the [B,1,S,S] bool mask exactly
        # as Dream's own _sample does (logical-and of the key/query keep-masks).
        m2d = full_attention_mask.bool()
        attn_mask_4d = m2d.unsqueeze(1).unsqueeze(-2) & m2d.unsqueeze(1).unsqueeze(-1)
        final_logits = model(
            x, attention_mask=attn_mask_4d
        ).logits[:, prompt_len:, :].float()
        del x, full_attention_mask, attn_mask_4d, m2d, attention_mask

        log_probs = F.log_softmax(final_logits, dim=-1)
        del final_logits
        k = min(top_k, log_probs.shape[-1])
        tk_logprobs, tk_ids = torch.topk(log_probs, k=k, dim=-1)
        del log_probs
        topk_logprobs_batches.append(tk_logprobs.cpu().to(torch.float32))
        topk_ids_batches.append(tk_ids.cpu().to(torch.int32))
        del tk_logprobs, tk_ids

        all_answers.extend(
            ans.strip()
            for ans in tokenizer.batch_decode(
                response_ids, skip_special_tokens=True
            )
        )
        del response_ids
        if is_cuda:
            torch.cuda.empty_cache()

    topk_data = {
        "topk_logprobs": torch.cat(topk_logprobs_batches, dim=0),
        "topk_token_ids": torch.cat(topk_ids_batches, dim=0),
    }
    return all_answers, topk_data
