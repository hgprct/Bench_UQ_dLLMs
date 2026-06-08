#!/bin/bash
# Submit GPU benchmarks for all available cluster nodes and report results.
#
# Usage:
#   bash scripts/submit_node_benchmarks.sh            # submit benchmark jobs
#   bash scripts/submit_node_benchmarks.sh --report   # print ranked report
#   bash scripts/submit_node_benchmarks.sh --report --results_dir /path/to/dir
#
# Results land in $STAGE_DIR/logs/node_benchmarks/<nodename>.json.
# Slurm logs:      $STAGE_DIR/logs/node_benchmarks/<nodename>_<jobid>.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Locate slurm_env.sh to resolve STAGE_DIR
# ---------------------------------------------------------------------------
source "$SCRIPT_DIR/slurm_env.sh"
RESULTS_DIR="${STAGE_DIR}/logs/node_benchmarks"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE="submit"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)  MODE="report" ;;
    --results_dir) RESULTS_DIR="$2"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Report mode: no GPU, no container needed — runs directly on login node
# ---------------------------------------------------------------------------
if [[ "$MODE" == "report" ]]; then
  echo "Reading results from: $RESULTS_DIR"
  python3 "$SCRIPT_DIR/bench_gpu.py" report --results_dir "$RESULTS_DIR"
  exit 0
fi

# ---------------------------------------------------------------------------
# Submit mode
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"

# Get all nodes in the default partition that are not down/drain/fail
NODES=$(sinfo -h -N -o "%N %t" -p defq 2>/dev/null \
  | awk '$2 !~ /^(down|drain|fail)/ {print $1}' \
  | sort -u)

if [[ -z "$NODES" ]]; then
  echo "ERROR: no usable nodes found in partition 'defq'" >&2
  exit 1
fi

echo "Submitting benchmark jobs..."
echo "Results dir: $RESULTS_DIR"
echo ""

SUBMITTED=0
SKIPPED=0
for NODE in $NODES; do
  # Skip if result already exists and is recent (< 7 days old)
  JSON="${RESULTS_DIR}/${NODE}.json"
  if [[ -f "$JSON" ]] && find "$JSON" -mtime -7 -print -quit 2>/dev/null | grep -q .; then
    echo "  [skip] $NODE  (result exists: $JSON)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  JOB_ID=$(sbatch \
    --nodelist="$NODE" \
    --export=ALL,PROJECT_PATH="$PROJECT_DIR" \
    "$SCRIPT_DIR/benchmark_node.slurm" \
    | awk '{print $NF}')
  echo "  [submit] $NODE  → job $JOB_ID"
  SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Submitted: $SUBMITTED  Skipped (cached): $SKIPPED"
echo ""
echo "Monitor progress:"
echo "  squeue -u \"\$USER\" -n gpu_bench"
echo ""
echo "When all jobs finish, run:"
echo "  bash scripts/submit_node_benchmarks.sh --report"
