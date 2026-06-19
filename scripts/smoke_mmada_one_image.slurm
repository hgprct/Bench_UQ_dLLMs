#!/bin/bash
#SBATCH --job-name=smoke_1img
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail
_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env
print_uq_header "Smoke test: MMaDA with one image"

run_in_uq_container gpu python scripts/smoke_mmada_one_image.py "$@"

echo "Finished at: $(date)"
