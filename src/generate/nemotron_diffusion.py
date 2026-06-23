"""Dedicated Nemotron-Labs-Diffusion-14B (text) inference.

NVIDIA's Nemotron-Labs-Diffusion-14B is a block-diffusion LLM exposing three
generation modes through ``trust_remote_code``::

    model.ar_generate(prompt_ids, ...)            # autoregressive
    model.generate(prompt_ids, ...)               # diffusion (dLM)  <-- used here
    model.linear_spec_generate(prompt_ids, ...)   # self-speculation

We use the diffusion mode. Unlike LLaDA, it is *threshold-driven*: tokens are
unmasked within a block once their confidence exceeds ``threshold``, so there is
no fixed ``steps`` count and no ``temperature``/``remasking`` knob -- the only
denoising controls are ``block_length`` and ``threshold``. ``model.generate``
returns ``(out_ids, nfe)`` where ``out_ids`` is the full prompt+answer sequence.

DEPENDENCY NOTE: Nemotron's custom code requires ``transformers>=5.0`` -- run it
in the separate "nemotron" environment (requirements_nemotron.in), not the
default 4.x env used by LLaDA / MMaDA / Dream.

Generation is one prompt at a time (batch_size is accepted but ignored): the
diffusion loop stops per-sequence on its own confidence schedule and the public
``model.generate`` shows no padded-batch / attention-mask contract, so per-item
is the safe choice (mirrors the multimodal paths). Batched decoding is a possible
later optimisation if the model is confirmed to support left-padded batches.

A single post-hoc forward pass over the completed sequence yields top-k log-probs,
returning the same ``(answers, topk_data)`` contract as the other backends.

Public API:
  generate(model, prompts, device, *, tokenizer, ...) -> (answers, topk_data)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _logits_of(output: Any) -> torch.Tensor:
    """Return the logits tensor from a forward pass.

    Nemotron loads via ``AutoModel`` + ``trust_remote_code``; its ``forward``
    returns either a ModelOutput with ``.logits`` or a raw logits tensor. This is
    the single integration point to adjust if a future revision differs.
    """
    return getattr(output, "logits", output)


def stack_ragged_topk(tensors: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    """Right-pad a list of ``[1, L_i, k]`` top-k tensors to a common length and
    stack to ``[N, Lmax, k]``.

    Nemotron generates one prompt at a time and the answer region
    (``out_ids[:, prompt_len:]``) has a *different* length per prompt -- prompts
    differ in length, and the model pads the whole sequence to a fixed total, so
    longer prompts leave a shorter answer span. The per-prompt top-k tensors
    therefore disagree on their sequence dim and cannot be ``torch.cat``-ed
    directly. Pad each to the longest answer span in this call. Padded positions
    are placeholders (no real token); downstream UQ should mask by answer length
    if it consumes them.
    """
    if not tensors:
        return torch.empty(0)
    max_len = max(t.shape[1] for t in tensors)
    out = []
    for t in tensors:
        if t.shape[1] < max_len:
            pad = t.new_full((t.shape[0], max_len - t.shape[1], t.shape[2]), pad_value)
            t = torch.cat([t, pad], dim=1)
        out.append(t)
    return torch.cat(out, dim=0)


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
    mask_id: int | None = None,
    eos_token_ids: list[int] | tuple[int, ...] | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    batch_size: int = 1,
    top_k: int = 64,
    # Nemotron-native knob: per-block confidence threshold for unmasking.
    threshold: float = 0.9,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Generate answers for text prompts using Nemotron diffusion ``model.generate``.

    ``steps``/``temperature``/``cfg_scale``/``remasking``/``mask_id`` and the
    EOS-logit knobs are accepted for interface parity but inert: Nemotron's
    diffusion loop is threshold-driven (see module docstring). Returns the same
    ``(answers, topk_data)`` contract.
    """
    block_length = int(block_size) if block_size else 32
    eos_id = (
        eos_token_ids[0]
        if eos_token_ids
        else getattr(tokenizer, "eos_token_id", None)
    )
    is_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    all_answers: list[str] = []
    topk_logprobs_batches: list[torch.Tensor] = []
    topk_ids_batches: list[torch.Tensor] = []

    for prompt in tqdm(prompts, total=len(prompts), desc="Generating (nemotron)"):
        input_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt",
        ).input_ids.to(device)
        prompt_len = input_ids.shape[1]

        out = model.generate(
            input_ids,
            max_new_tokens=max_gen_length,
            block_length=block_length,
            threshold=threshold,
            eos_token_id=eos_id,
        )
        out_ids = out[0] if isinstance(out, (tuple, list)) else out
        response_ids = out_ids[:, prompt_len:]

        final_logits = _logits_of(model(out_ids))[:, prompt_len:, :].float()
        log_probs = F.log_softmax(final_logits, dim=-1)
        del final_logits
        k = min(top_k, log_probs.shape[-1])
        tk_logprobs, tk_ids = torch.topk(log_probs, k=k, dim=-1)
        del log_probs
        topk_logprobs_batches.append(tk_logprobs.cpu().to(torch.float32))
        topk_ids_batches.append(tk_ids.cpu().to(torch.int32))

        all_answers.append(
            tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
        )
        del input_ids, out, out_ids, response_ids, tk_logprobs, tk_ids
        if is_cuda:
            torch.cuda.empty_cache()

    topk_data = {
        "topk_logprobs": stack_ragged_topk(topk_logprobs_batches, 0.0),
        "topk_token_ids": stack_ragged_topk(topk_ids_batches, 0.0),
    }
    return all_answers, topk_data
