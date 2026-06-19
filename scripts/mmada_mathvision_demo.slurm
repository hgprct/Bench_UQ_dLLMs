#!/bin/bash
#SBATCH --job-name=mmada_mathvision
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
print_uq_header "MMaDA x MathVision visual smoke test"

run_in_uq_container gpu python scripts/mmada_mathvision_demo.py "$@"

echo "Finished at: $(date)"
