"""NLI entailment model loading and inference.

Wraps microsoft/deberta-v2-xlarge-mnli (or compatible models) for
semantic equivalence detection used by sampling and trajectory features.
"""

from __future__ import annotations

from typing import Any

DEFAULT_NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"


def load_entailment_model(
    model_name: str = DEFAULT_NLI_MODEL,
    *,
    device: str | None = None,
) -> Any:
    """Load an NLI entailment model.

    Supports the project's pre-cached deberta model and any compatible
    HuggingFace NLI model.
    """
    from src.semantic._deberta import EntailmentDeberta
    return EntailmentDeberta(model_name=model_name, device=device)
