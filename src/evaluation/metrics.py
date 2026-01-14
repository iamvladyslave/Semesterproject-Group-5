from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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
    exact_match = float((preds == targets).all(dim=1).float().mean().item())

    precision = tp.float() / (tp + fp + EPS)
    recall = tp.float() / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    macro = {
        "accuracy": float(per_concept_acc.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
        "exact_match": exact_match,
    }
    return {
        "per_concept_accuracy": per_concept_acc.tolist(),
        "per_concept_precision": precision.tolist(),
        "per_concept_recall": recall.tolist(),
        "per_concept_f1": f1.tolist(),
        "macro": macro,
        "exact_match": exact_match,
    }


def label_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, object]:
    """
    Compute classification accuracy, per-class precision/recall/F1, and confusion matrix.
    """
    preds = torch.argmax(logits, dim=1)
    targets = targets.long()
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    accuracy = float((preds == targets).float().mean().item())
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np,
        preds_np,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    cm = confusion_matrix(targets_np, preds_np, labels=list(range(num_classes)))

    macro = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    return {
        "accuracy": accuracy,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "macro": macro,
    }
