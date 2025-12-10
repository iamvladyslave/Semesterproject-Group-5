from __future__ import annotations

from typing import Dict, List, Tuple

import torch

EPS = 1e-8


def _counts(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tp = ((preds == 1) & (targets == 1)).sum(dim=0)
    fp = ((preds == 1) & (targets == 0)).sum(dim=0)
    fn = ((preds == 0) & (targets == 1)).sum(dim=0)
    return tp, fp, fn


def concept_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, List[float]]:
    """
    Compute per-concept accuracy, precision, recall, and F1.
    Returns a dict with per-concept lists and macro averages.
    """
    preds = (torch.sigmoid(logits) >= threshold).int()
    targets = targets.int()

    per_concept_acc = (preds == targets).float().mean(dim=0)
    tp, fp, fn = _counts(preds, targets)

    precision = tp.float() / (tp + fp + EPS)
    recall = tp.float() / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    macro = {
        "accuracy": float(per_concept_acc.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
    }
    return {
        "per_concept_accuracy": per_concept_acc.tolist(),
        "per_concept_precision": precision.tolist(),
        "per_concept_recall": recall.tolist(),
        "per_concept_f1": f1.tolist(),
        "macro": macro,
    }
