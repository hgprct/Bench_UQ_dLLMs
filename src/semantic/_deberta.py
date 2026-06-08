"""DeBERTa NLI model wrapper -- ported from semantic_unc/entailment.py."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class EntailmentDeberta:
    """NLI inference with DeBERTa-v2-xlarge-mnli.

    Label mapping: 0=contradiction, 1=neutral, 2=entailment.
    """

    entailment_id = 2
    contradiction_id = 0
    neutral_id = 1

    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli", device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

    def batch_probabilities(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run NLI inference and return (probabilities, predicted_classes)."""
        import torch

        from tqdm import tqdm

        all_probs = []
        all_classes = []
        for i in tqdm(range(0, len(premises), batch_size), desc="NLI inference"):
            batch_p = list(premises[i:i + batch_size])
            batch_h = list(hypotheses[i:i + batch_size])
            inputs = self.tokenizer(
                batch_p, batch_h, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            classes = probs.argmax(axis=-1)
            all_probs.append(probs)
            all_classes.append(classes)

        if not all_probs:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int64)
        return np.vstack(all_probs), np.concatenate(all_classes)
