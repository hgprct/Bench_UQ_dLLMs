"""Smoke-test that every declared dependency can be imported."""

import subprocess
import sys
import pytest


DIRECT_IMPORTS = [
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


def _check_importable(module: str) -> None:
    # Run in a fresh subprocess so conftest's src/ sys.path injection
    # does not shadow installed packages (e.g. src/datasets/ vs HuggingFace datasets).
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("pip_name,module", DIRECT_IMPORTS, ids=[m for _, m in DIRECT_IMPORTS])
def test_direct_import(pip_name, module):
    _check_importable(module)


@pytest.mark.parametrize("pip_name,module", TRANSITIVE_IMPORTS, ids=[m for _, m in TRANSITIVE_IMPORTS])
def test_transitive_import(pip_name, module):
    _check_importable(module)


def test_comet_import():
    pytest.importorskip("comet")
