"""LLaDA diffusion generation with block-diffusion support.

Based on the official LLaDA inference code (ML-GSAI/LLaDA).
Supports semi-autoregressive generation via block_size < max_gen_length.

Public API:
  generate(model, prompts, device, *, tokenizer, ...) -> (answers, topk_logprobs)
  calibrate_batch_size(model, tokenizer, device, ...) -> int
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    mask_index: torch.Tensor, steps: int
) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def calibrate_batch_size(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    prompts: list[str],
    max_gen_length: int,
    mask_id: int,
    max_batch_size: int = 64,
    safety_factor: float = 0.9,
) -> int:
    """Find the largest batch size that fits in GPU memory via binary search.

    Runs real forward passes at the worst-case (longest) prompt length to find
    the maximum batch size, then applies *safety_factor* to leave headroom for
    the denoising loop overhead (confidence tensors, Gumbel noise, etc.).
    """
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return max_batch_size

    encoded_lengths = [
        len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts
    ]
    max_prompt_len = max(encoded_lengths)
    total_seq_len = max_prompt_len + max_gen_length

    lo, hi, best = 1, max_batch_size, 1

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            dummy = torch.full(
                (mid, total_seq_len), mask_id, dtype=torch.long, device=device,
            )
            attn = torch.ones_like(dummy)
            out = model(dummy, attention_mask=attn).logits
            out_gen = out[:, max_prompt_len:, :].float()
            _ = F.log_softmax(out_gen, dim=-1)
            del dummy, attn, out, out_gen, _
            torch.cuda.empty_cache()
            best = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            del dummy, attn
            torch.cuda.empty_cache()
            hi = mid - 1

    calibrated = max(1, int(best * safety_factor))
    print(f"[calibrate] max_prompt_len={max_prompt_len}  seq_len={total_seq_len}  "
          f"raw_max_bs={best}  calibrated_bs={calibrated}")
    return calibrated


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
    mask_id: int = 126336,
    eos_token_ids: list[int] | tuple[int, ...] | None = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    batch_size: int = 8,
    top_k: int = 64,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Generate answers for a list of prompts using LLaDA diffusion.

    Returns
    -------
    answers : list[str]
        Decoded text answers, one per prompt.
    topk_data : dict[str, Tensor]
        "topk_logprobs": (N, max_gen_length, K) float32
        "topk_token_ids": (N, max_gen_length, K) int32
    """
    if block_size is None:
        block_size = max_gen_length

    assert max_gen_length % block_size == 0, (
        f"max_gen_length ({max_gen_length}) must be divisible by block_size ({block_size})"
    )

    all_answers: list[str] = []
    topk_logprobs_batches: list[torch.Tensor] = []
    topk_ids_batches: list[torch.Tensor] = []
    is_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    for batch_start in tqdm(
        range(0, len(prompts), batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc="Generating",
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

        x = torch.full(
            (input_ids.shape[0], prompt_len + max_gen_length),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        x[:, :prompt_len] = input_ids
        del input_ids

        full_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (x.shape[0], max_gen_length),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )
        del attention_mask

        prompt_index = x != mask_id

        num_blocks = max_gen_length // block_size
        steps_per_block = steps // num_blocks

        for num_block in range(num_blocks):
            block_start_pos = prompt_len + num_block * block_size
            block_end = prompt_len + (num_block + 1) * block_size
            block_mask_index = x[:, block_start_pos:block_end] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for step_i in range(steps_per_block):
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    attn_ = torch.cat(
                        [full_attention_mask, full_attention_mask], dim=0
                    )
                    logits = model(x_, attention_mask=attn_).logits
                    del x_, attn_, un_x
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    del un_logits
                else:
                    logits = model(x, attention_mask=full_attention_mask).logits

                if logits_eos_inf and eos_token_ids:
                    for eos_id in eos_token_ids:
                        if 0 <= eos_id < logits.shape[-1]:
                            logits[:, :, eos_id] = -torch.inf

                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                del logits_with_noise

                if remasking == "low_confidence":
                    if confidence_eos_eot_inf and eos_token_ids:
                        for eos_id in eos_token_ids:
                            if 0 <= eos_id < logits.shape[-1]:
                                logits[:, :, eos_id] = -torch.inf
                    p = F.softmax(logits.float(), dim=-1)
                    del logits
                    x0_p = torch.gather(
                        p, dim=-1, index=x0.unsqueeze(-1)
                    ).squeeze(-1)
                    del p
                elif remasking == "random":
                    del logits
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise ValueError(f"Unsupported remasking: {remasking}")

                x0_p[:, block_end:] = -torch.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                del x0_p

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, sel = torch.topk(
                        confidence[j], k=int(num_transfer_tokens[j, step_i])
                    )
                    transfer_index[j, sel] = True
                x[transfer_index] = x0[transfer_index]
                del x0, confidence, transfer_index, mask_index

        del prompt_index

        if is_cuda:
            torch.cuda.empty_cache()

        response_ids = x[:, prompt_len:]

        final_logits = model(
            x, attention_mask=full_attention_mask
        ).logits[:, prompt_len:, :]
        del x, full_attention_mask

        final_logits = final_logits.float()
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
