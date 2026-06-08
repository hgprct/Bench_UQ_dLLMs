#!/bin/bash
#SBATCH --job-name=gpu_bench
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/node_benchmarks/%N_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/node_benchmarks/%N_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Single-node GPU benchmark job. Submitted by submit_node_benchmarks.sh with
# --nodelist=<node> so each invocation is pinned to one specific node.
#
# Results are written to $STAGE_DIR/logs/node_benchmarks/<nodename>.json.
# Run the report with: bash scripts/submit_node_benchmarks.sh --report

set -euo pipefail
_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"
init_uq_slurm_env

NODE=$(hostname -s)
OUT_DIR="${STAGE_DIR}/logs/node_benchmarks"
OUT_JSON="${OUT_DIR}/${NODE}.json"

print_uq_header "GPU Node Benchmark: $NODE"

run_in_uq_container gpu \
  python scripts/bench_gpu.py bench \
    --output "$OUT_JSON" \
    --gpu 0

echo
echo "Benchmark complete: $OUT_JSON"
