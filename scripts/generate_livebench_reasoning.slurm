#!/bin/bash
#SBATCH --job-name=uq_livebench_reason
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=dalianvl03

set -euo pipefail

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env

CONFIG="configs/LLaDA1.5_livebench_reasoning_l256_s256_lc_b32.json"
RUN_ID="$(basename "$CONFIG" .json)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}/$RUN_ID"

print_uq_header "Generate LiveBench Reasoning: $CONFIG -> $OUTPUT_DIR"

run_in_uq_container gpu python -m src.cli.generate \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR" \
  --num_questions 10

echo
echo "Generation finished at: $(date)"
