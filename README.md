# uncertainty-DLM

Uncertainty quantification pipeline for discrete diffusion language models. The pipeline generates text with masked-diffusion denoising (with optional block-diffusion), labels correctness, computes uncertainty quantification (UQ) features, and evaluates their discriminative power with AUROC, AUPRC, PRR, and other metrics. A separate K-fold cross-validation pipeline compares learned step selection against baselines.

## Project structure

```
src/
  config.py              Enums, per-dataset defaults, config builder
  registry.py            Dataset module registry
  seed.py                Global RNG seeding
  split.py               Train/val/test + K-fold splitting
  kfold.py               K-fold aggregation (mean/std across folds)
  generate/              Denoising loop, model loading, trace serialization, batch size calibration
  datasets/              Dataset adapters (triviaqa, gsm8k, wmt14_fr_en, xsum, samsum, hotpotqa, musique)
  features/              UQ feature computation (token-level + NLI-based)
  evaluate/              Metrics (AUROC, AUPRC, PRR, ECE, Brier) + bootstrap CI
  semantic/              NLI entailment model (DeBERTa-v2-xlarge-mnli)
  judge/                 LLM judge for correctness labeling (vLLM backend)
  labeling/              Labeling dispatch (exact_match or llm_judge)
  utils/                 JSON/JSONL I/O + XLSX workbook export
  cli/                   CLI entry points for each stage
configs/                 JSON config files (<Model>_<dataset>_l<max_gen_length>_s<steps>_<remasking>.json)
scripts/                 SLURM job scripts and shell helpers
tests/                   Pytest test suite
```

## Canonical names

No aliases. Use exactly these names everywhere:

- **Models**: `LLaDA`, `LLaDA1.5`
- **Datasets**: `triviaqa`, `gsm8k`, `wmt14_fr_en`, `xsum`, `samsum`, `hotpotqa`, `musique`
- **Remasking**: `lc` (low confidence), `rd` (random)

## Per-dataset defaults

Each dataset has its own generation defaults defined in `DATASET_CONFIGS` (`src/config.py`):

| Dataset       | max_gen_length | steps | block_size | fewshot_k |
|---------------|---------------|-------|------------|-----------|
| triviaqa      | 128           | 128   | None       | 0         |
| gsm8k         | 256           | 128   | None       | 4         |
| wmt14_fr_en   | 128           | 128   | None       | 0         |
| xsum          | 128           | 128   | None       | 0         |
| samsum        | 128           | 128   | None       | 0         |
| hotpotqa      | 128           | 128   | None       | 0         |
| musique       | 128           | 128   | None       | 0         |

These defaults are used by `build_generation_config()` and `gen_configs` when no explicit override is given.

## Generation

Generation uses a masked-diffusion denoising loop with support for block-diffusion (semi-autoregressive generation where `max_gen_length` is split into blocks of `block_size`). When `block_size` is `None`, the full generation length is denoised in a single pass.

Default generation produces 1 greedy + 20 stochastic (T=1.0) answers per prompt.

### Adaptive batch size

When `--auto_batch_size` is enabled (the default), the generation CLI runs a binary search with real forward passes to find the largest batch size that fits in GPU memory, then applies a 0.9 safety factor. Explicitly passing `--batch_size N` disables auto-calibration.

### Prompt preparation

Prompts can be pre-built on a CPU node and saved as `prompts.jsonl` files, so the GPU generation job skips dataset loading:

```bash
python scripts/prepare_all_prompts.py --output-dir outputs configs/*.json
```

## Pipelines

### 1. Performance evaluation (AUROC/PRR)

```
generate -> label -> split -> features -> evaluate -> export
```

| Stage    | CLI module         | GPU   | Description                                           |
|----------|--------------------|-------|-------------------------------------------------------|
| generate | `src.cli.generate` | yes   | Denoising loop + top-k logprob extraction             |
| label    | `src.cli.label`    | varies| Label greedy answers correct/incorrect                |
| split    | `src.cli.split`    | no    | Train/val/test split (default 10/30/40)               |
| features | `src.cli.features` | yes   | Compute UQ features (msp, perplexity, mte, mcnse, semantic entropy, etc.) |
| evaluate | `src.cli.evaluate` | no    | AUROC/AUPRC/PRR/ECE/Brier with bootstrap CI           |
| export   | `src.cli.export`   | no    | XLSX workbook export                                  |

### 2. K-fold cross-validation (step selection vs baseline)

After generation + labeling:

```
baseline -> kfold -> report
```

| Stage    | CLI module         | GPU | Description                                                              |
|----------|--------------------|-----|--------------------------------------------------------------------------|
| baseline | `src.cli.baseline` | yes | Compute token + iid-sample + full-trajectory features for all prompts    |
| kfold    | `src.cli.kfold`    | yes | K folds: grid-search QP weights on train+val, evaluate on test           |
| report   | `src.cli.report`   | no  | XLSX workbook + bar chart plots (baseline vs selection with error bars)   |

### Common helpers

- `src.cli.gen_configs` -- generate config files programmatically
- `src.cli.pipeline` -- orchestrate Pipeline 1 stages in sequence
- `src.cli.label_all` -- batch-label all run directories
- `src.cli.build_prompts` -- build and save prompts for a config

## Running on DALIA (HPC)

All Python execution goes through SLURM + Apptainer. **Never run Python directly on the login node.**

### Environment setup

`scripts/slurm_env.sh` provides three shell functions used by all SLURM scripts:
- `init_uq_slurm_env` -- sets `$IMAGE`, `$PROJECT_DIR`, creates log dirs
- `print_uq_header` -- prints job metadata
- `run_in_uq_container {gpu|cpu} <command>` -- runs a command inside the Apptainer container

A `.env` file at the project root (see `.env.example`) provides `HF_TOKEN`, `STAGE_DIR_PATH`, `PROJECT_PATH`, etc. It is automatically sourced by each SLURM script.

### Full generation across all datasets

The three-phase launch script handles config generation, prompt building, and SLURM array submission:

```bash
bash scripts/launch_llada15_gen.sh [options]
```

Options:
- `--concurrency N` -- max simultaneous SLURM tasks (default: 6)
- `--output-dir DIR` -- parent output directory (default: outputs)
- `--dry-run` -- print commands without submitting
- `--prompts-only` -- only build prompts, don't submit generation jobs
- `--num-questions N` -- override number of questions per dataset (default: 1000)

Phase 1 generates config JSONs using per-dataset defaults. Phase 2 builds `prompts.jsonl` files inside the container (CPU). Phase 3 submits a SLURM array job (`scripts/generate_array.slurm`) for GPU generation.

### Pipeline 1: full performance pipeline

```bash
sbatch scripts/pipeline_v2.slurm <model> <dataset> <num_response_samples> <length> <steps> <remasking>
```

Example:
```bash
sbatch scripts/pipeline_v2.slurm LLaDA triviaqa 20 128 64 lc
```

Environment overrides (set before sbatch or export):
```bash
NUM_QUESTIONS=500 TEMPERATURE=0.8 BATCH_SIZE=8 SEED=42 \
LABEL_METHOD=exact_match JUDGE_MODEL=meta-llama/Llama-3.3-70B-Instruct \
NLI_MODEL=microsoft/deberta-v2-xlarge-mnli BOOTSTRAP_SAMPLES=1000 \
  sbatch scripts/pipeline_v2.slurm LLaDA gsm8k 20 256 128 rd
```

### Pipeline 1: individual stages

```bash
sbatch scripts/generate_v2.slurm --config configs/LLaDA1.5_triviaqa_l128_s128_lc.json
sbatch scripts/label_v2.slurm outputs/LLaDA1.5_triviaqa_l128_s128_lc/
sbatch scripts/features_v2.slurm --input_logs outputs/LLaDA1.5_triviaqa_l128_s128_lc/ \
  --output outputs/LLaDA1.5_triviaqa_l128_s128_lc/uq_features.jsonl
sbatch scripts/evaluate_v2.slurm --features_path outputs/LLaDA1.5_triviaqa_l128_s128_lc/uq_features.jsonl \
  --output_metrics_json outputs/LLaDA1.5_triviaqa_l128_s128_lc/uq_eval_metrics.json
```

Run via the Python orchestrator:
```bash
python -m src.cli.pipeline --model LLaDA --dataset triviaqa \
  --length 128 --steps 64 --remasking lc --stages generate label
```

### Pipeline 2: K-fold cross-validation

**Phase 1 -- baseline features:**
```bash
sbatch scripts/baseline_v2.slurm outputs/LLaDA1.5_triviaqa_l128_s128_lc
```

**Phase 2 -- K folds with per-fold grid search:**
```bash
sbatch scripts/kfold_v2.slurm outputs/LLaDA1.5_triviaqa_l128_s128_lc 10 \
  outputs/LLaDA1.5_triviaqa_l128_s128_lc/baseline_features.jsonl
```

**Phase 3 -- XLSX + bar chart report:**
```bash
OUTPUTS_ROOT=outputs/LLaDA1.5_triviaqa_l128_s128_lc \
OUTPUT_DIR=outputs/LLaDA1.5_triviaqa_l128_s128_lc/kfold/report \
  sbatch scripts/report_v2.slurm
```

### Generate config files

```bash
# Uses per-dataset max_gen_length/steps/fewshot_k from DATASET_CONFIGS
python -m src.cli.gen_configs --model LLaDA1.5 --dataset triviaqa gsm8k \
  --remasking lc rd

# Explicit overrides
python -m src.cli.gen_configs --model LLaDA1.5 --dataset triviaqa \
  --max-gen-length 64 --steps 64 --block-size 32 --remasking lc
```

### Preflight checks

```bash
bash scripts/preflight.sh          # import check + GPU benchmark
bash scripts/preflight.sh imports  # import check only
bash scripts/preflight.sh submit   # submit to all cluster nodes
bash scripts/preflight.sh report   # print GPU benchmark report
```

### Tests

```bash
sbatch scripts/tests_v2.slurm
TEST_TARGET="tests/test_config.py" sbatch scripts/tests_v2.slurm
PYTEST_ARGS="-q -x" sbatch scripts/tests_v2.slurm
```

## Config files

Config JSON files follow the naming convention `<Model>_<dataset>_l<max_gen_length>_s<steps>_<remasking>[_b<block_size>][_fs<fewshot_k>].json`.

Example (`configs/LLaDA1.5_gsm8k_l256_s128_lc_fs4.json`):
```json
{
  "model_family": "DLM",
  "model_id": "GSAI-ML/LLaDA-1.5",
  "dataset": "gsm8k",
  "max_gen_length": 256,
  "steps": 128,
  "block_size": null,
  "remasking": "lc",
  "temperature": 1.0,
  "num_response_samples": 20,
  "generate_greedy": true,
  "fewshot_k": 4,
  "num_questions": 1000
}
```

## Outputs

Each run writes to `outputs/<run_id>/`:

| File | Stage | Content |
|------|-------|---------|
| `config.json` | generate | Resolved generation config |
| `examples.jsonl` | generate + label | One record per QA sample; `final.is_correct` after labeling |
| `answers.jsonl` | generate | Prompt/question/answer triplets |
| `traces.npz` | generate | Dense tensor arrays (token IDs, logprobs, top-k) |
| `metadata.json` | generate | Run manifest and file index |
| `prompts.jsonl` | prepare | Pre-built prompts (optional, built in Phase 2 of launch script) |
| `splits.json` | split | Train/val/test prompt ID split |
| `uq_features.jsonl` | features | One row per prompt with all UQ features |
| `uq_eval_metrics.{json,csv}` | evaluate | AUROC/AUPRC/PRR/ECE/Brier with bootstrap CI |
| `uq_results.xlsx` | export | Formatted workbook with color-coded performance |
| `baseline_features.jsonl` | baseline | All token + iid-sample + full-trajectory features |
| `kfold/kfold_summary.{json,csv}` | kfold | Mean/std of metrics across folds |
| `kfold/fold_<i>/` | kfold | Per-fold grid-search weights + metrics |
| `kfold/report/kfold_results.xlsx` | report | Comparison workbook |
| `kfold/report/kfold_<metric>.pdf` | report | Bar chart with error bars |

## Reproducibility

- All RNGs seeded via `src.seed.seed_everything(seed)`
- Dependencies managed with `uv` (`pyproject.toml`)
- Two-pass generation: seed for greedy, seed+1 for sampled
- Deterministic train/val/test split with configurable absolute counts
- K-fold splits deterministic from `(seed, k, n_train)`
