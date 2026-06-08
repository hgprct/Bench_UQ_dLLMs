# uncertainty-DLM

Uncertainty quantification pipeline for discrete diffusion language models (LLaDA, LLaDA1.5, Dream). The pipeline generates text with masked-diffusion denoising, captures per-step traces, labels correctness, computes uncertainty quantification (UQ) features, and evaluates their discriminative power with AUROC, AUPRC, PRR, and other metrics. A separate K-fold cross-validation pipeline compares learned step selection against baselines, and a standalone latency profiler measures per-feature wall-clock overhead for Pareto-curve comparisons.

## Project structure

```
src/
  config.py              Enums, HF IDs, config builder
  registry.py            Dataset module registry
  seed.py                Global RNG seeding
  split.py               Train/val/test + K-fold splitting
  kfold.py               K-fold aggregation (mean/std across folds)
  generate/              Denoising loop, model loading, trace serialization
  datasets/              Dataset adapters (triviaqa, gsm8k, wmt14_fr_en, wmt14_de_en, xsum, samsum)
  features/              UQ feature computation (token-level + NLI-based)
  evaluate/              Metrics (AUROC, AUPRC, PRR, ECE, Brier) + bootstrap CI
  semantic/              NLI entailment model (DeBERTa-v2-xlarge-mnli)
  judge/                 LLM judge for correctness labeling (vLLM backend)
  mmd/                   MMD infrastructure (embeddings, kernels, scopes)
  selection/             QP-learned temporal step selection
  timing/                Latency profiler (per-feature wall-clock overhead)
  io/                    JSON/JSONL I/O + XLSX workbook export
  cli/                   CLI entry points for each stage
configs/                 JSON config files (<Model>_<dataset>_l<length>_s<steps>_<remasking>.json)
scripts/                 Slurm job scripts and shell helpers
tests/                   Pytest test suite
```

## Canonical names

No aliases. Use exactly these names everywhere:

- **Models**: `LLaDA`, `LLaDA1.5`, `Dream`
- **Datasets**: `triviaqa`, `gsm8k`, `wmt14_fr_en`, `wmt14_de_en`, `xsum`, `samsum`
- **Remasking**: `lc` (low confidence), `rd` (random)

## Pipelines

The project ships three independent pipelines on top of a shared `generate -> label` foundation.

### 1. Performance evaluation (AUROC/PRR)

```
generate -> label -> split -> features -> evaluate -> export
```

| Stage | CLI module | GPU | Description |
|-------|-----------|-----|-------------|
| generate | `src.cli.generate` | yes | Denoising loop + trace capture |
| label | `src.cli.label` | varies | Label greedy answers correct/incorrect |
| split | `src.cli.split` | no | Train/val/test split (default 10/30/40) |
| features | `src.cli.features` | yes | Compute UQ features (msp, perplexity, mte, mcnse, semantic entropy, etc.) |
| evaluate | `src.cli.evaluate` | no | AUROC/AUPRC/PRR/ECE/Brier with bootstrap CI |
| export | `src.cli.export` | no | XLSX workbook export |

### 2. K-fold cross-validation (step selection vs baseline)

After generation + labeling, three independent phases:

```
baseline -> kfold -> report
```

| Stage | CLI module | GPU | Description |
|-------|-----------|-----|-------------|
| baseline | `src.cli.baseline` | yes | Compute token + iid-sample + full-trajectory features for **all** prompts |
| kfold | `src.cli.kfold` | yes | K folds, per fold: grid-search QP weights on train+val, evaluate baseline + selection on test |
| report | `src.cli.report` | no | XLSX workbook + bar chart plots (baseline vs selection with error bars) |

### 3. Latency profiler (per-feature wall-clock overhead)

Standalone, fully independent of pipelines 1 and 2. Reuses the existing generation and NLI code.

| CLI module | GPU | Description |
|-----------|-----|-------------|
| `src.cli.profile_latency` | yes | Per-feature wall-clock latency on N fresh prompts (default 50). Outputs CSV summary + raw JSONL + metadata |

### Common helpers

- `src.cli.select {train,evaluate,grid_search}` — standalone QP step-selection commands
- `src.cli.gen_configs` — generate config files programmatically
- `src.cli.pipeline` — orchestrate Pipeline 1 stages in sequence

## Running on DALIA (HPC)

All Python execution goes through Slurm + Apptainer. **Never run Python directly on the login node.**

### Environment setup

`scripts/slurm_env.sh` provides three shell functions used by all Slurm scripts:
- `init_uq_slurm_env` -- sets `$IMAGE`, `$PROJECT_DIR`, creates log dirs
- `print_uq_header` -- prints job metadata
- `run_in_uq_container {gpu|cpu} <command>` -- runs a command inside the Apptainer container

A `.env` file at the project root (see `.env.example`) provides `HF_TOKEN`, `STAGE_DIR_PATH`, `PROJECT_PATH`, etc. It is automatically sourced by each Slurm script.

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
  sbatch scripts/pipeline_v2.slurm LLaDA gsm8k 20 128 64 rd
```

Sweep all configs in `configs/`:
```bash
bash scripts/run_all_configs_v2.sh 20
```

### Pipeline 1: individual stages

```bash
sbatch scripts/generate_v2.slurm --config configs/LLaDA_triviaqa_l128_s64_lc.json
sbatch scripts/label_v2.slurm outputs/LLaDA_triviaqa_l128_s64_lc/
sbatch scripts/features_v2.slurm --input_logs outputs/LLaDA_triviaqa_l128_s64_lc/ \
  --output outputs/LLaDA_triviaqa_l128_s64_lc/uq_features.jsonl
sbatch scripts/evaluate_v2.slurm --features_path outputs/LLaDA_triviaqa_l128_s64_lc/uq_features.jsonl \
  --output_metrics_json outputs/LLaDA_triviaqa_l128_s64_lc/uq_eval_metrics.json
```

Run a subset via the Python orchestrator:
```bash
python -m src.cli.pipeline --model LLaDA --dataset triviaqa \
  --length 128 --steps 64 --remasking lc --stages generate label
```

### Pipeline 2: K-fold cross-validation

**Phase 1 — baseline features (one job per run dir):**
```bash
sbatch scripts/baseline_v2.slurm outputs/LLaDA_triviaqa_l128_s64_lc
```
Output: `<run_dir>/baseline_features.jsonl`.

**Phase 2 — K folds with per-fold grid search:**
```bash
sbatch scripts/kfold_v2.slurm outputs/LLaDA_triviaqa_l128_s64_lc 10 \
  outputs/LLaDA_triviaqa_l128_s64_lc/baseline_features.jsonl
```
Args: `<run_dir> <n_train> <baseline_features_path>`. Outputs to `<run_dir>/kfold/`: per-fold dirs, `kfold_summary.json`, `kfold_summary.csv`.

Direct CLI form (full control):
```bash
python -m src.cli.kfold \
  --run_dir outputs/LLaDA_triviaqa_l128_s64_lc \
  --baseline_features_path outputs/LLaDA_triviaqa_l128_s64_lc/baseline_features.jsonl \
  --n_train 10 --n_folds 5 --seed 42 \
  --lambda_values 0.001 0.01 0.1 1.0 10.0 --budget_k 8
```

**Phase 3 — XLSX + bar chart report:**
```bash
OUTPUTS_ROOT=outputs/LLaDA_triviaqa_l128_s64_lc \
OUTPUT_DIR=outputs/LLaDA_triviaqa_l128_s64_lc/kfold/report \
  sbatch scripts/report_v2.slurm
```
Outputs to `<kfold_dir>/report/`: `kfold_results.xlsx`, `kfold_<metric>.pdf`.

### Pipeline 3: latency profiler

```bash
python -m src.cli.profile_latency \
  --config configs/LLaDA_triviaqa_l128_s64_lc.json \
  --output_dir outputs/LLaDA_triviaqa_l128_s64_lc/latency \
  --num_prompts 50 --warmup_prompts 2 \
  --selection_dir outputs/selection/LLaDA_triviaqa_l128_s64_lc \
  --budget_k 8 \
  --seed 42
```

The profiler generates fresh prompts from the HF dataset (no dependency on any prior run for prompt selection), times every UQ feature per prompt under the standalone-cost attribution model, then averages across prompts.

Outputs to `--output_dir`:
- `latency_summary.csv` — one row per `(run_id, family, feature)` with `time_{gen,nli,math,total}_{mean,std}`.
- `latency_raw.jsonl` — per-`(prompt, feature)` measurements for re-aggregation or CI plots.
- `latency_metadata.json` — config snapshot, autodetected NLI batch size, greedy/extras generation baselines.

If `--selection_dir` points at a directory without `trained_weights.npz`, `selected-*` features are skipped (warning logged).

### Step selection (standalone)

```bash
# Train weights
python -m src.cli.select train --run_dir outputs/LLaDA_triviaqa_l128_s64_lc/ \
  --output_dir outputs/selection/ --n_train 10 --n_val 30 --n_test 40

# Evaluate with learned weights
python -m src.cli.select evaluate --run_dir outputs/LLaDA_triviaqa_l128_s64_lc/ \
  --weights_dir outputs/selection/ --discretization_method top_k

# Lambda grid search
python -m src.cli.select grid_search --run_dir outputs/LLaDA_triviaqa_l128_s64_lc/ \
  --output_dir outputs/grid/ --lambda_values 0.01 0.1 1.0 10.0 \
  --n_train 10 --n_val 30 --n_test 40
```

### Generate config files

```bash
python -m src.cli.gen_configs --model LLaDA Dream --dataset triviaqa gsm8k \
  --length 128 --remasking lc rd
```

### Relabel all runs

```bash
bash scripts/relabel_all.sh
```

### Tests

```bash
sbatch scripts/tests_v2.slurm
TEST_TARGET="tests/test_timing/" sbatch scripts/tests_v2.slurm
PYTEST_ARGS="-q -x" sbatch scripts/tests_v2.slurm
```

### Build the Apptainer container

```bash
sbatch build_container.slurm
```

## Outputs

Each run writes to `outputs/<run_id>/`:

| File | Pipeline | Content |
|------|----------|---------|
| `config.json` | gen | Resolved generation config |
| `examples.jsonl` | gen + label | One record per QA sample with per-step trace strings; `final.is_correct` after labeling |
| `answers.jsonl` | gen | Prompt/question/answer triplets |
| `traces.npz` | gen | Dense tensor arrays (token IDs, logprobs, mask status, top-k) |
| `metadata.json` | gen | Run manifest and file index |
| `splits.json` | split | Train/val/test prompt ID split |
| `uq_features.jsonl` | features | One row per prompt with all UQ features |
| `uq_eval_metrics.{json,csv}` | evaluate | AUROC/AUPRC/PRR/ECE/Brier with bootstrap CI |
| `uq_results.xlsx` | export | Formatted workbook with color-coded performance |
| `baseline_features.jsonl` | baseline | All token + iid-sample + full-trajectory features for every prompt |
| `kfold/kfold_summary.{json,csv}` | kfold | Mean/std of metrics across folds, per feature, baseline vs selection |
| `kfold/fold_<i>/` | kfold | Per-fold grid-search weights, baseline + selection metrics |
| `kfold/report/kfold_results.xlsx` | report | Comparison workbook |
| `kfold/report/kfold_<metric>.pdf` | report | Bar chart with error bars (baseline vs selection) |
| `latency/latency_summary.csv` | profile_latency | Per-feature wall-clock mean/std over the prompt sample |
| `latency/latency_raw.jsonl` | profile_latency | Per-`(prompt, feature)` raw measurements |
| `latency/latency_metadata.json` | profile_latency | Run-level diagnostics |

## Reproducibility

- All RNGs seeded via `src.seed.seed_everything(seed)`
- Dependencies managed with `uv` (`pyproject.toml`)
- Two-pass generation: seed for greedy, seed+1 for sampled
- Deterministic train/val/test split with configurable absolute counts
- K-fold splits deterministic from `(seed, k, n_train)`
- Latency profiler seeds the fresh HF prompt sample; per-prompt NLI cache is bypassed so each prompt pays its full N×N cost (no inter-prompt amortization)
