"""CLI entry point for generation: produce answers + top-k logprobs."""

from __future__ import annotations

import argparse
import json
import os

import torch

from src.generate import calibrate_batch_size, generate as generate_fn
from src.generate import dream, nemotron_diffusion, nemotron_vlm
from src.generate.mmada_mmu import generate_mmu
from src.generate.traces import save_results
from src.config import (
    BATCHING_FAMILIES, Dataset, GENERATION_DEFAULTS, MMADA_VQ_MODEL_ID,
    MULTIMODAL_FAMILIES, Model, Remasking, build_generation_config,
    derive_run_id, load_config, model_family, model_name_from_id,
    resolve_remasking,
)
from src.seed import seed_everything
from src.datasets.dataloader import build_raw_prompts, load_prompts_jsonl
from src.datasets.outputs import load_image, write_results_jsonl
from src.generate.inputs import (
    apply_chat_template,
    expand_greedy_and_sampled,
    expand_response_samples,
)
from src.generate.model import (
    infer_eos_token_ids, infer_mask_token_id,
    load_model, load_tokenizer, load_vq_model, pad_token_id as get_pad_token_id,
)


_CLI_OVERRIDE_KEYS = (
    "num_questions", "num_response_samples", "batch_size", "steps",
    "max_gen_length", "block_size", "temperature", "cfg_scale",
    "generate_greedy", "remasking", "mask_id",
    "top_k", "fewshot_k", "confidence_eos_eot_inf", "seed",
)


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if torch.cuda.device_count() == 1:
        return torch.device("cuda:0")
    best_device = 0
    best_free = 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_device = i
    return torch.device(f"cuda:{best_device}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dLLM answers + top-k logprobs.")
    src = parser.add_argument_group("config source (pick one)")
    src.add_argument("--config", help="Path to generation config JSON")
    src.add_argument("--model", choices=[m.value for m in Model],
                     help="Model name (builds config from registry)")
    src.add_argument("--dataset", choices=[d.value for d in Dataset],
                     help="Dataset name (builds config from registry)")
    parser.add_argument("--run_id", help="Run ID for output directory naming")
    parser.add_argument("--num_questions", type=int)
    parser.add_argument("--num_response_samples", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max_gen_length", type=int)
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--cfg_scale", type=float)
    parser.add_argument("--generate_greedy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--remasking", choices=["lc", "rd"])
    parser.add_argument("--mask_id", type=int)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--fewshot_k", type=int)
    parser.add_argument("--confidence_eos_eot_inf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--auto_batch_size", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-calibrate batch size to fit GPU memory (default: True). "
                             "Disabled when --batch_size is explicitly set.")
    parser.add_argument("--prompts", help="Path to prompts.jsonl (from prepare stage)")
    parser.add_argument("--output_dir")
    args = parser.parse_args(argv)
    if not args.config and not (args.model and args.dataset):
        parser.error("Provide either --config or both --model and --dataset")
    if args.config and (args.model or args.dataset):
        parser.error("--config and --model/--dataset are mutually exclusive")
    return args


def _apply_config(config: dict, args: argparse.Namespace) -> None:
    for key in _CLI_OVERRIDE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
    if args.run_id:
        config["run_id"] = args.run_id
    for key, default in GENERATION_DEFAULTS.items():
        config.setdefault(key, default)


def _setup_model(config: dict, device):
    model_id = config["model_id"]
    model = load_model(model_id, device)
    tokenizer = load_tokenizer(model_id)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    # MMaDA multimodal-understanding needs the MAGVIT-v2 VQ image tokenizer.
    vq_model = None
    if model_family(model_id) == "mmada":
        vq_id = config.get("vq_model_id", MMADA_VQ_MODEL_ID)
        print(f"[gen] Loading VQ image tokenizer: {vq_id}")
        vq_model = load_vq_model(vq_id, device)

    eos_token_ids = infer_eos_token_ids(tokenizer, model_id=model_id)
    ptid = get_pad_token_id(tokenizer)
    if ptid is None:
        if eos_token_ids:
            tokenizer.pad_token_id = eos_token_ids[0]
            ptid = tokenizer.pad_token_id
            print(f"[gen] No pad_token_id; using EOS id {ptid}")
        else:
            raise ValueError("No pad_token_id and no EOS tokens available")

    if config.get("mask_id") is None:
        config["mask_id"] = infer_mask_token_id(tokenizer, model=model)
    # Only the block-diffusion backends (LLaDA, MMaDA, Dream) consume an explicit
    # mask id. Nemotron's threshold-driven loop handles masking internally.
    if config.get("mask_id") is None and model_family(model_id) in (
        "llada", "mmada", "dream",
    ):
        raise ValueError("mask_id could not be inferred; set it in the config")
    config["eos_token_ids"] = eos_token_ids

    return model, tokenizer, eos_token_ids, ptid, vq_model


def _load_or_prepare_inputs(config, args, tokenizer, output_dir, multimodal_family):
    """Return ``(dataset_key, qa_pairs, prompts, raw_prompts)``.

    ``prompts`` are chat-templated (text backends tokenize these directly);
    ``raw_prompts`` are untemplated. The multimodal backends build their own
    chat/message structure around the image, so they consume raw text -- and
    their tokenizers may not even define a chat template -- hence chat-templating
    is skipped for multimodal families (``prompts`` mirrors ``raw_prompts``).
    """
    prompts_path = args.prompts or os.path.join(output_dir, "prompts.jsonl")
    if os.path.isfile(prompts_path):
        print(f"Loading prepared prompts from {prompts_path}...")
        dataset_key, qa_pairs, raw_prompts = load_prompts_jsonl(prompts_path)
        print(f"Loaded {len(qa_pairs)} prompts from '{dataset_key}'.")
        # prompts.jsonl is text-only (images are not serialised). For multimodal
        # backends, reattach each image from the shared on-disk images folder.
        if multimodal_family:
            missing = 0
            for qa in qa_pairs:
                if qa.get("image") is None:
                    img = load_image(dataset_key, str(qa.get("image_id") or qa.get("id")))
                    if img is None:
                        missing += 1
                    else:
                        qa["image"] = img
            if missing:
                print(
                    f"[gen] WARNING: {missing}/{len(qa_pairs)} images missing under "
                    f"outputs/{dataset_key}/images/ (re-run data prep for this dataset)"
                )
    else:
        fewshot_k = int(config["fewshot_k"])
        if fewshot_k > 0:
            print(f"[gen] Few-shot: {fewshot_k} examples")
        hf_token = os.environ.get("HF_TOKEN", "")
        print("Preparing dataset inputs inline...")
        dataset_key, qa_pairs, raw_prompts = build_raw_prompts(config, hf_token)
        if not qa_pairs:
            raise ValueError("No QA pairs found in dataset")
        print(f"Prepared {len(qa_pairs)} prompts from '{dataset_key}'.")
    prompts = raw_prompts if multimodal_family else apply_chat_template(raw_prompts, tokenizer)
    return dataset_key, qa_pairs, prompts, raw_prompts


def _run_generation(qa_pairs, prompts, config, gen_one, seed):
    """Drive greedy / sampled generation.

    *gen_one(prompts, qa_list, temperature)* runs one generation pass and
    returns ``(answers, topk_data)``. The qa_list is passed in lockstep with
    prompts so the multimodal path can recover per-prompt images from it; the
    text path ignores it.

    Returns ``(merged_qa, merged_prompts, all_answers, topk_data,
    greedy_by_prompt)`` where ``greedy_by_prompt`` holds the single greedy answer
    per *original* prompt (``None`` when no greedy pass was run), used to fill the
    clean per-model results JSONL.
    """
    num_sampled = int(config["num_response_samples"])
    temperature = float(config["temperature"])
    generate_greedy = bool(config["generate_greedy"])

    if temperature > 0:
        if num_sampled <= 0:
            raise ValueError("num_response_samples must be > 0 when temperature > 0")

        if generate_greedy:
            total = 1 + num_sampled
            print(f"[gen] Two-pass: 1 greedy + {num_sampled} sampled (T={temperature})")

            greedy_answers, greedy_topk = gen_one(prompts, qa_pairs, 0.0)

            seed_everything(seed + 1)
            sampled_qa, sampled_prompts = expand_response_samples(qa_pairs, prompts, num_sampled)
            sampled_answers, sampled_topk = gen_one(sampled_prompts, sampled_qa, temperature)

            merged_qa, merged_prompts = expand_greedy_and_sampled(qa_pairs, prompts, num_sampled)
            all_answers = _interleave_answers(greedy_answers, sampled_answers, len(qa_pairs), num_sampled)
            topk_data = _interleave_topk(greedy_topk, sampled_topk, len(qa_pairs), num_sampled)
            config["num_response_samples"] = total
            greedy_by_prompt = list(greedy_answers)
        else:
            print(f"[gen] Sampled-only: {num_sampled} samples per prompt (T={temperature})")
            merged_qa, merged_prompts = expand_response_samples(qa_pairs, prompts, num_sampled)
            all_answers, topk_data = gen_one(merged_prompts, merged_qa, temperature)
            greedy_by_prompt = [None] * len(qa_pairs)
    else:
        if num_sampled > 1:
            print(f"[gen] temperature=0: ignoring num_response_samples={num_sampled}, generating 1 greedy response per prompt")
        print("[gen] Greedy-only: 1 deterministic response per prompt")
        merged_qa, merged_prompts = list(qa_pairs), list(prompts)
        all_answers, topk_data = gen_one(merged_prompts, merged_qa, 0.0)
        config["num_response_samples"] = 1
        greedy_by_prompt = list(all_answers)

    return merged_qa, merged_prompts, all_answers, topk_data, greedy_by_prompt


def _interleave_answers(
    greedy_answers: list[str],
    sampled_answers: list[str],
    num_questions: int,
    num_sampled: int,
) -> list[str]:
    merged = []
    for q in range(num_questions):
        merged.append(greedy_answers[q])
        merged.extend(sampled_answers[q * num_sampled : (q + 1) * num_sampled])
    return merged


def _interleave_topk(
    greedy_topk: dict[str, torch.Tensor],
    sampled_topk: dict[str, torch.Tensor],
    num_questions: int,
    num_sampled: int,
) -> dict[str, torch.Tensor]:
    interleaved = {}
    for key in sorted(greedy_topk):
        gt = greedy_topk[key]
        st = sampled_topk[key]
        chunks = []
        for q in range(num_questions):
            chunks.append(gt[q : q + 1])
            chunks.append(st[q * num_sampled : (q + 1) * num_sampled])
        interleaved[key] = torch.cat(chunks, dim=0)
    return interleaved


def _resolve_config(args: argparse.Namespace) -> dict:
    if args.config:
        return load_config(args.config)
    return build_generation_config(
        Model(args.model),
        Dataset(args.dataset),
        args.max_gen_length,
        args.steps,
        Remasking(args.remasking or GENERATION_DEFAULTS["remasking"]),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _resolve_config(args)
    _apply_config(config, args)

    seed = int(config["seed"])
    seed_everything(seed)

    run_id = str(config.get("run_id") or derive_run_id(config))
    output_dir = args.output_dir or config.get("output_dir") or os.path.join("outputs", run_id)
    config["run_id"] = run_id

    try:
        device = select_device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] Device: {device}")
    print(f"[gen] Model: {config['model_id']}")

    family = model_family(config["model_id"])
    print(f"[gen] Inference family: {family}")

    model, tokenizer, eos_token_ids, ptid, vq_model = _setup_model(config, device)
    dataset_key, qa_pairs, prompts, raw_prompts = _load_or_prepare_inputs(
        config, args, tokenizer, output_dir,
        multimodal_family=family in MULTIMODAL_FAMILIES,
    )

    # Multimodal backends (MMaDA mmu, Nemotron-VLM) are taken when the model is a
    # vision model and the dataset carries images (e.g. MathVision). They build
    # their own chat/message structure around the image, so they consume the raw
    # (untemplated) prompt text; text backends use the chat-templated prompts.
    has_images = bool(qa_pairs) and isinstance(qa_pairs[0], dict) and qa_pairs[0].get("image") is not None
    multimodal = family in MULTIMODAL_FAMILIES and has_images
    if family in MULTIMODAL_FAMILIES and not has_images:
        raise ValueError(
            f"Model '{config['model_id']}' is multimodal ({family}) but dataset "
            f"'{dataset_key}' provides no images; this model runs via the "
            "image+text path only."
        )
    gen_prompts = raw_prompts if multimodal else prompts

    block_size = config.get("block_size")
    if block_size is not None:
        block_size = int(block_size)

    # Auto-calibrate batch size only for the batched text backends (LLaDA, Dream)
    # and only when the user did not set --batch_size. The multimodal and Nemotron
    # backends generate one item at a time (variable-length sequences / per-item
    # threshold loops), so calibration -- which probes batched text forward passes
    # -- is skipped and batch_size is pinned to 1.
    use_auto_bs = (
        args.auto_batch_size and args.batch_size is None
        and family in BATCHING_FAMILIES
    )
    if multimodal or family == "nemotron":
        config["batch_size"] = 1
        print(f"[gen] {family}: per-item generation (batch_size=1)")
    elif use_auto_bs:
        print("[gen] Calibrating batch size...")
        calibrated_bs = calibrate_batch_size(
            model, tokenizer, device,
            prompts=prompts,
            max_gen_length=config["max_gen_length"],
            mask_id=config["mask_id"],
            max_batch_size=64,
        )
        config["batch_size"] = calibrated_bs
        print(f"[gen] Using auto-calibrated batch_size={calibrated_bs}")
    else:
        print(f"[gen] Using configured batch_size={config['batch_size']}")

    # Common kwargs shared by the block-diffusion text backends. mask_id may be
    # None for Nemotron (threshold-driven); its module ignores it.
    common_kwargs = dict(
        model=model,
        device=device,
        batch_size=config["batch_size"],
        tokenizer=tokenizer,
        steps=config["steps"],
        max_gen_length=config["max_gen_length"],
        block_size=block_size,
        cfg_scale=float(config.get("cfg_scale", 0.0)),
        remasking=resolve_remasking(config["remasking"]),
        mask_id=config.get("mask_id"),
        eos_token_ids=eos_token_ids,
        logits_eos_inf=config["logits_eos_inf"],
        confidence_eos_eot_inf=config["confidence_eos_eot_inf"],
        top_k=config["top_k"],
    )
    threshold = float(config.get("nemotron_threshold", 0.9))

    def gen_one(call_prompts, qa_list, temperature):
        if family == "mmada":
            images = [qa["image"] for qa in qa_list]
            return generate_mmu(
                model, call_prompts, images, device,
                tokenizer=tokenizer, vq_model=vq_model,
                steps=config["steps"], max_gen_length=config["max_gen_length"],
                block_size=block_size, temperature=temperature,
                cfg_scale=float(config.get("cfg_scale", 0.0)),
                remasking=resolve_remasking(config["remasking"]),
                mask_id=config["mask_id"], top_k=config["top_k"],
            )
        if family == "nemotron_vlm":
            images = [qa["image"] for qa in qa_list]
            return nemotron_vlm.generate_vlm(
                model, call_prompts, images, device,
                tokenizer=tokenizer, model_id=config["model_id"],
                steps=config["steps"], max_gen_length=config["max_gen_length"],
                block_size=block_size, temperature=temperature,
                top_k=config["top_k"], threshold=threshold,
            )
        if family == "dream":
            return dream.generate(
                prompts=call_prompts, temperature=temperature,
                alg=config.get("dream_alg"),
                alg_temp=config.get("dream_alg_temp", 0.0),
                dream_top_p=config.get("dream_top_p"),
                dream_top_k=config.get("dream_top_k"),
                **common_kwargs,
            )
        if family == "nemotron":
            return nemotron_diffusion.generate(
                prompts=call_prompts, temperature=temperature,
                threshold=threshold, **common_kwargs,
            )
        return generate_fn(prompts=call_prompts, temperature=temperature, **common_kwargs)

    merged_qa, merged_prompts, all_answers, topk_data, greedy_by_prompt = _run_generation(
        qa_pairs, gen_prompts, config, gen_one, seed,
    )

    # PIL images cannot be JSON-serialised; drop them before writing traces.
    merged_qa = [
        {k: v for k, v in qa.items() if k != "image"} if isinstance(qa, dict) else qa
        for qa in merged_qa
    ]

    from src.registry import get_dataset_module
    dataset_module = get_dataset_module(dataset_key)
    parse_answer_fn = getattr(dataset_module, "parse_answer", None)

    summary = save_results(
        output_dir=output_dir,
        run_config=config,
        qa_pairs=merged_qa,
        prompts=merged_prompts,
        answers=all_answers,
        topk_data=topk_data,
        dataset_key=dataset_key,
        parse_answer=parse_answer_fn,
    )

    # Clean per-model results JSONL: one record per original prompt with the
    # canonical QASample fields, greedy_answer + model_name filled in. Written to
    # the shared per-dataset folder (outputs/<dataset>/<model_name>.jsonl),
    # alongside the per-run UQ traces above.
    model_name = model_name_from_id(config["model_id"])
    result_samples = []
    for i, qa in enumerate(qa_pairs):
        sample = {k: v for k, v in qa.items() if k != "image"}
        sample["greedy_answer"] = greedy_by_prompt[i] if i < len(greedy_by_prompt) else None
        sample["model_name"] = model_name
        result_samples.append(sample)
    clean_path = write_results_jsonl(dataset_key, model_name, result_samples)

    print(f"\nSaved to: {output_dir}")
    print(f"Clean results: {clean_path}")
    print(f"Examples: {summary['num_examples']}")


if __name__ == "__main__":
    main()
