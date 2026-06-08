#!/bin/bash
#SBATCH --job-name=check_imports
#SBATCH --output=/lustre/work/pdl16831/%u/stage/logs/%x_%j.out
#SBATCH --error=/lustre/work/pdl16831/%u/stage/logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$_dir/.env" ] && { set -a; source "$_dir/.env"; set +a; }
source "${PROJECT_PATH:-$_dir}/scripts/slurm_env.sh"

init_uq_slurm_env
print_uq_header "Import check for all dependencies"

run_in_uq_container gpu python -c '
import sys

checks = [
    # requirements.in
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"),
    ("openpyxl", "openpyxl"),
    ("packaging", "packaging"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("sentence-transformers", "sentence_transformers"),
    ("cvxpy", "cvxpy"),
    ("accelerate", "accelerate"),
    ("huggingface-hub", "huggingface_hub"),
    ("sentencepiece", "sentencepiece"),
    ("tokenizers", "tokenizers"),
    ("safetensors", "safetensors"),
    ("regex", "regex"),
    ("dill", "dill"),
    ("xxhash", "xxhash"),
    ("multiprocess", "multiprocess"),
    ("python-dotenv", "dotenv"),
    ("pytest", "pytest"),
    ("ruff", "ruff"),
    # container packages
    ("torch", "torch"),
    ("vllm", "vllm"),
    ("matplotlib", "matplotlib"),
]

failed = []
for name, module in checks:
    try:
        m = __import__(module)
        version = getattr(m, "__version__", "?")
        print(f"  OK  {name:25s} -> {module} ({version})")
    except ImportError as e:
        print(f"  FAIL {name:25s} -> {e}")
        failed.append(name)

print()
if failed:
    msg = ", ".join(failed)
    print(f"FAILED imports ({len(failed)}): {msg}")
    sys.exit(1)
else:
    print(f"All {len(checks)} imports OK")
'

echo "Finished at: $(date)"
