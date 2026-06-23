#!/bin/bash
#SBATCH --job-name=uq_smoke
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=dalianvl02,dalianvl03,dalianvl04,dalianvl05
#
# Smoke-test ONE model on a couple of prompts of every dataset available to it.
# One SLURM job per model (submit via scripts/launch_smoke_all.sh).
#
# Each (model, dataset) pair runs src.cli.generate greedily (temperature 0 ->
# one deterministic response per prompt) on NUM_QUESTIONS prompts and writes the
# raw generation traces. NOTHING is judged or scored: answers.jsonl already
# carries the raw model output plus all the QA fields (question, reference
# answer, prompt, id, ...). The LLM-judge / evaluate stages are deliberately NOT
# invoked. (answers.jsonl also has a passive `parsed_answer` column from the
# dataset's own light parser -- not a judge; ignore it if you only want
# `raw_answer`.)
#
# Required env:
#   MODEL           one of LLaDA / LLaDA1.5 / Dream / Nemotron / MMaDA / NemotronVLM
# Optional env:
#   NUM_QUESTIONS   prompts per dataset (default: 2)
#   OUTBASE         output root (default: $PROJECT_DIR/outputs/smoke_all)
#
# Dataset routing: text models get the text datasets; the vision models (MMaDA,
# NemotronVLM) get the image datasets (the CLI hard-refuses a VLM on a text-only
# dataset, and a text model cannot see images). The two Nemotron families need
# the transformers-4.57.1 overlay (set automatically here).

set -uo pipefail   # NOTE: no -e; we want to continue past a failing dataset.

MODEL="${MODEL:?Set MODEL=<LLaDA1.5|Dream|Nemotron|MMaDA|NemotronVLM>}"

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
_proj="${PROJECT_PATH:-$_dir}"

# ── Family routing: overlay + dataset list ────────────────────────────────
case "$MODEL" in
  Nemotron|NemotronVLM)
    # transformers-4.57.1 layer (see model-inference-backends notes).
    export CONTAINER_OVERLAY="${CONTAINER_OVERLAY:-$_proj/overlay_nemotron.img}"
    ;;
esac

case "$MODEL" in
  MMaDA|NemotronVLM)
    DATASETS=(mathvision vqa_rad drive_bench vilp)
    ;;
  LLaDA|LLaDA1.5|Dream|Nemotron)
    DATASETS=(triviaqa gsm8k wmt14_fr_en xsum samsum hotpotqa musique \
              competition_math champ medmcqa livebench_reasoning)
    ;;
  *)
    echo "ERROR: unknown MODEL='$MODEL'" >&2
    exit 1
    ;;
esac

source "$_proj/scripts/slurm_env.sh"
init_uq_slurm_env

NUM_QUESTIONS="${NUM_QUESTIONS:-2}"
OUTBASE="${OUTBASE:-$PROJECT_DIR/outputs/smoke_all}"
[[ "$OUTBASE" != /* ]] && OUTBASE="$PROJECT_DIR/$OUTBASE"
mkdir -p "$OUTBASE/$MODEL"

print_uq_header "Smoke: $MODEL on ${#DATASETS[@]} datasets x ${NUM_QUESTIONS} prompts (raw answers only)"
echo "Overlay:  ${CONTAINER_OVERLAY:-<none>}"
echo "Outbase:  $OUTBASE/$MODEL"
echo "Datasets: ${DATASETS[*]}"

declare -a OK_DS=() FAIL_DS=()

for ds in "${DATASETS[@]}"; do
  outdir="$OUTBASE/$MODEL/$ds"
  echo
  echo "############################################################"
  echo "# $MODEL :: $ds  ->  $outdir"
  echo "# $(date)"
  echo "############################################################"

  # --batch_size 2 skips auto-calibration for the 2-prompt smoke (the vision /
  # Nemotron backends pin batch_size=1 internally regardless). --temperature 0
  # forces a single greedy response per prompt (overrides the build-config
  # default of T=1, which would otherwise draw 20 samples).
  if run_in_uq_container gpu python -m src.cli.generate \
        --model "$MODEL" \
        --dataset "$ds" \
        --num_questions "$NUM_QUESTIONS" \
        --temperature 0 \
        --batch_size 2 \
        --output_dir "$outdir"; then
    OK_DS+=("$ds")
  else
    echo "!!! FAILED: $MODEL :: $ds (continuing)"
    FAIL_DS+=("$ds")
  fi
done

# ── Per-model summary ─────────────────────────────────────────────────────
echo
echo "============================================================"
echo "SMOKE SUMMARY: $MODEL"
echo "============================================================"
for ds in "${OK_DS[@]:-}"; do
  [ -z "$ds" ] && continue
  af="$OUTBASE/$MODEL/$ds/answers.jsonl"
  n=0; [ -f "$af" ] && n=$(wc -l < "$af")
  printf "  OK    %-22s %s answers -> %s\n" "$ds" "$n" "$af"
done
for ds in "${FAIL_DS[@]:-}"; do
  [ -z "$ds" ] && continue
  printf "  FAIL  %-22s (see log above)\n" "$ds"
done
echo "------------------------------------------------------------"
echo "  ${#OK_DS[@]} ok / ${#FAIL_DS[@]} failed / ${#DATASETS[@]} total"
echo "Finished at: $(date)"
