#!/bin/bash
#SBATCH --job-name=build_prompts
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.txt
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j_err.txt
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env
print_uq_header "Build prompts check"

run_in_uq_container gpu python -m src.cli.build_prompts "$@"

echo "Finished at: $(date)"
