#!/bin/bash
#SBATCH --job-name=uq_prepare
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch scripts/prepare_v2.slurm --config configs/LLaDA_triviaqa_l32_s32_lc.json
#   sbatch scripts/prepare_v2.slurm --all
#   sbatch scripts/prepare_v2.slurm --all --overwrite
#
# Environment overrides:
#   OUTPUT_ROOT=outputs

set -euo pipefail
_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"
init_uq_slurm_env

PREPARE_ARGS=("$@")
[ -n "${OUTPUT_ROOT:-}" ] && PREPARE_ARGS+=(--output-root "$OUTPUT_ROOT")

print_uq_header "Prepare prompts"

run_in_uq_container cpu python -m src.cli.prepare "${PREPARE_ARGS[@]}"

echo
echo "Prepare finished at: $(date)"
