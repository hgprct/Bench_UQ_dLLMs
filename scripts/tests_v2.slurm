#!/bin/bash
#SBATCH --job-name=uq_tests
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

export TEST_TARGET="${TEST_TARGET:-tests/}"
export PYTEST_ARGS="${PYTEST_ARGS:--q -x}"

init_uq_slurm_env
print_uq_header "Bench_UQ_dLLMs test suite"
echo "Test target: $TEST_TARGET"
echo "Pytest args: $PYTEST_ARGS"

run_in_uq_container gpu bash -lc 'python -m pytest $PYTEST_ARGS $TEST_TARGET'
echo "Finished at: $(date)"
