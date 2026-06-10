"""Smoke-test that every declared dependency can be imported."""

import subprocess
import sys
import pytest


CORE_IMPORTS = [
    # (pip package name, python module to import)
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
    ("torchvision", "torchvision"),
]

TRANSITIVE_IMPORTS = [
    ("accelerate", "accelerate"),
    ("huggingface-hub", "huggingface_hub"),
    ("sentencepiece", "sentencepiece"),
    ("tokenizers", "tokenizers"),
    ("safetensors", "safetensors"),
    ("regex", "regex"),
    ("dill", "dill"),
    ("xxhash", "xxhash"),
    ("multiprocess", "multiprocess"),
]

GPU_IMPORTS = [
    ("torch", "torch"),
    ("vllm", "vllm"),
    ("matplotlib", "matplotlib"),
]

DEV_IMPORTS = [
    ("pytest", "pytest"),
    ("ruff", "ruff"),
]


def _check_importable(module: str) -> None:
    # Run in a fresh subprocess so conftest's src/ sys.path injection
    # does not shadow installed packages (e.g. src/datasets/ vs HuggingFace datasets).
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("pip_name,module", CORE_IMPORTS, ids=[m for _, m in CORE_IMPORTS])
def test_core_import(pip_name, module):
    _check_importable(module)


@pytest.mark.parametrize("pip_name,module", TRANSITIVE_IMPORTS, ids=[m for _, m in TRANSITIVE_IMPORTS])
def test_transitive_import(pip_name, module):
    _check_importable(module)


@pytest.mark.parametrize("pip_name,module", GPU_IMPORTS, ids=[m for _, m in GPU_IMPORTS])
def test_gpu_import(pip_name, module):
    _check_importable(module)


@pytest.mark.parametrize("pip_name,module", DEV_IMPORTS, ids=[m for _, m in DEV_IMPORTS])
def test_dev_import(pip_name, module):
    _check_importable(module)
