#!/bin/bash
# Unified preflight script: import check + GPU benchmark.
#
# Usage:
#   bash scripts/preflight.sh [MODE]
#
# Modes:
#   all      (default) import check + GPU benchmark on this node
#   imports  import check only
#   bench    GPU benchmark on this node
#   submit   submit preflight jobs to all cluster nodes
#   report   print GPU benchmark report (no GPU needed)
#
# GPU-needing modes (all, imports, bench) auto-submit via sbatch when
# run from a login node (i.e. outside a SLURM allocation).

#SBATCH --job-name=preflight
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Resolve the real project root (BASH_SOURCE is unreliable inside SLURM
# spool, so prefer PROJECT_PATH from .env / --export, then SLURM_SUBMIT_DIR).
_guess_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$_guess_dir/.env" ] && { set -a; source "$_guess_dir/.env"; set +a; }
export PROJECT_DIR="${PROJECT_PATH:-$_guess_dir}"
SCRIPT_DIR="$PROJECT_DIR/scripts"

source "$SCRIPT_DIR/slurm_env.sh"

RESULTS_DIR="${STAGE_DIR}/logs/node_benchmarks"
MODE="${1:-all}"

# ── Auto-submit to compute node if needed ────────────────────────────────
needs_gpu() {
  case "$1" in all|imports|bench) return 0 ;; *) return 1 ;; esac
}

if needs_gpu "$MODE" && [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Not inside a SLURM job — submitting to a compute node..."
  JOB_ID=$(sbatch \
    --export=ALL,PROJECT_PATH="$PROJECT_DIR" \
    "$SCRIPT_DIR/preflight.sh" "$MODE" \
    | awk '{print $NF}')
  echo "Submitted job $JOB_ID  (mode: $MODE)"
  echo ""
  echo "Monitor:  squeue -u \$USER -n preflight"
  echo "Logs:     tail -f ${STAGE_DIR}/logs/preflight_${JOB_ID}.out"
  exit 0
fi

# ── Import check ─────────────────────────────────────────────────────────
do_imports() {
  init_uq_slurm_env
  print_uq_header "Import check for all dependencies"

  run_in_uq_container gpu python -c '
import sys

# Matches pyproject.toml [project].dependencies
core = [
    ("numpy",               "numpy"),
    ("scipy",               "scipy"),
    ("scikit-learn",        "sklearn"),
    ("tqdm",                "tqdm"),
    ("openpyxl",            "openpyxl"),
    ("packaging",           "packaging"),
    ("transformers",        "transformers"),
    ("datasets",            "datasets"),
    ("sentence-transformers","sentence_transformers"),
    ("cvxpy",               "cvxpy"),
    ("torchvision",         "torchvision"),
]

# Transitive deps required in --no-deps install environment
transitive = [
    ("accelerate",      "accelerate"),
    ("huggingface-hub", "huggingface_hub"),
    ("sentencepiece",   "sentencepiece"),
    ("tokenizers",      "tokenizers"),
    ("safetensors",     "safetensors"),
    ("regex",           "regex"),
    ("dill",            "dill"),
    ("xxhash",          "xxhash"),
    ("multiprocess",    "multiprocess"),
]

# GPU / container packages ([project.optional-dependencies])
gpu = [
    ("torch",       "torch"),
    ("vllm",        "vllm"),
    ("matplotlib",  "matplotlib"),
]

# Dev tools
dev = [
    ("pytest", "pytest"),
    ("ruff",   "ruff"),
]

failed = []
for section_name, checks in [
    ("Core", core),
    ("Transitive", transitive),
    ("GPU / container", gpu),
    ("Dev", dev),
]:
    print(f"\n── {section_name} ──")
    for name, module in checks:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "?")
            print(f"  OK   {name:25s} -> {module} ({version})")
        except ImportError as e:
            print(f"  FAIL {name:25s} -> {e}")
            failed.append(name)

print()
if failed:
    msg = ", ".join(failed)
    print(f"FAILED imports ({len(failed)}): {msg}")
    sys.exit(1)
else:
    total = len(core) + len(transitive) + len(gpu) + len(dev)
    print(f"All {total} imports OK")
'

  echo "Import check finished at: $(date)"
}

# ── GPU benchmark (single node) ─────────────────────────────────────────
do_bench() {
  init_uq_slurm_env

  local node
  node=$(hostname -s)
  local out_dir="$RESULTS_DIR"
  local out_json="${out_dir}/${node}.json"
  mkdir -p "$out_dir"

  print_uq_header "GPU Node Benchmark: $node"

  run_in_uq_container gpu \
    python scripts/bench_gpu.py bench \
      --output "$out_json" \
      --gpu 0

  echo
  echo "Benchmark complete: $out_json"
}

# ── Submit benchmarks to all cluster nodes ───────────────────────────────
do_submit() {
  mkdir -p "$RESULTS_DIR"

  local nodes
  nodes=$(sinfo -h -N -o "%N %t" -p defq 2>/dev/null \
    | awk '$2 !~ /^(down|drain|fail)/ {print $1}' \
    | sort -u)

  if [[ -z "$nodes" ]]; then
    echo "ERROR: no usable nodes found in partition 'defq'" >&2
    exit 1
  fi

  echo "Submitting preflight jobs (imports + bench)..."
  echo "Results dir: $RESULTS_DIR"
  echo ""

  local submitted=0 skipped=0
  for node in $nodes; do
    local json="${RESULTS_DIR}/${node}.json"
    if [[ -f "$json" ]] && find "$json" -mtime -7 -print -quit 2>/dev/null | grep -q .; then
      echo "  [skip] $node  (result < 7 days old)"
      skipped=$((skipped + 1))
      continue
    fi

    local job_id
    job_id=$(sbatch \
      --nodelist="$node" \
      --export=ALL,PROJECT_PATH="$PROJECT_DIR" \
      "$SCRIPT_DIR/preflight.sh" all \
      | awk '{print $NF}')
    echo "  [submit] $node  -> job $job_id"
    submitted=$((submitted + 1))
  done

  echo ""
  echo "Submitted: $submitted  Skipped (cached): $skipped"
  echo ""
  echo "Monitor:  squeue -u \"\$USER\" -n preflight"
  echo "Report:   bash scripts/preflight.sh report"
}

# ── Print benchmark report ───────────────────────────────────────────────
do_report() {
  echo "Reading results from: $RESULTS_DIR"
  python3 "$SCRIPT_DIR/bench_gpu.py" report --results_dir "$RESULTS_DIR"
}

# ── Dispatch ─────────────────────────────────────────────────────────────
case "$MODE" in
  imports)
    do_imports
    ;;
  bench)
    do_bench
    ;;
  submit)
    do_submit
    ;;
  report)
    do_report
    ;;
  all)
    do_imports
    echo ""
    do_bench
    ;;
  *)
    echo "Usage: $0 {all|imports|bench|submit|report}" >&2
    exit 1
    ;;
esac
