"""QA pair definition.

All fields are optional. Each dataset populates the fields it supports
and consumers check for what they need.
"""

from __future__ import annotations

from typing import Any, TypedDict


class QASample(TypedDict, total=False):
    """Canonical in-memory QA row passed from adapters into the pipeline.

    The first nine fields are the canonical, serialisable contract written to the
    per-model results JSONL (``src.datasets.outputs.write_results_jsonl``).

    ``image`` is a runtime-only convenience field: multimodal adapters attach the
    decoded ``PIL.Image`` here so the generation backends can consume it. It is
    never serialised -- it is dropped before any JSONL is written and the image is
    instead persisted once to the shared ``outputs/<dataset>/images/`` folder,
    keyed by ``image_id``.
    """
    id: str
    image_id: str
    full_prompt: str
    question: str
    ground_truth_answer: Any
    greedy_answer: str | None
    model_name: str | None
    dataset_name: str
    split: str
    image: Any
