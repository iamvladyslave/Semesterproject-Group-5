from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import matplotlib.pyplot as plt


def plot_training_curves(history: Dict[str, List[float]], *, save_path: Optional[str] = None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(history.get("train_loss", []), label="Train")
    ax[0].plot(history.get("val_loss", []), label="Val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history.get("train_acc", []), label="Train")
    ax[1].plot(history.get("val_acc", []), label="Val")
    ax[1].plot(history.get("train_vec_acc", []), label="Train (exact-match)")
    ax[1].plot(history.get("val_vec_acc", []), label="Val (exact-match)")
    ax[1].set_title("Concept Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_per_concept_performance(
    values: List[float],
    *,
    title: str = "Per-Concept Metric",
    concept_names: Optional[List[str]] = None,
    ylabel: str = "Accuracy",
    save_path: Optional[str] = None,
):
    idx = list(range(len(values)))
    labels = concept_names if concept_names is not None else [f"C{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(idx, values)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    *,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None,
):
    cm = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    labels = class_names if class_names is not None else [str(i) for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_example_predictions(
    images: torch.Tensor,
    *,
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    concept_targets: torch.Tensor,
    concept_preds: torch.Tensor,
    concept_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    max_concepts: Optional[int] = None,
    save_path: Optional[str] = None,
):
    images = images.detach().cpu()
    concept_targets = concept_targets.detach().cpu()
    concept_preds = concept_preds.detach().cpu()

    n = min(len(true_labels), images.shape[0])
    if n == 0:
        return None

    num_concepts = concept_targets.shape[1]
    indices = list(range(num_concepts))
    if max_concepts is not None and max_concepts < num_concepts:
        indices = indices[:max_concepts]
    labels = concept_names if concept_names is not None else [f"C{i}" for i in indices]

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        true_lbl = true_labels[i]
        pred_lbl = pred_labels[i]
        true_name = class_names[true_lbl] if class_names is not None else str(true_lbl)
        pred_name = class_names[pred_lbl] if class_names is not None else str(pred_lbl)
        axes[i, 0].set_title(f"True: {true_name} | Pred: {pred_name}")

        target_vals = concept_targets[i][indices].numpy()
        pred_vals = concept_preds[i][indices].numpy()
        x = np.arange(len(indices))
        axes[i, 1].bar(x - 0.2, target_vals, width=0.4, label="True")
        axes[i, 1].bar(x + 0.2, pred_vals, width=0.4, label="Pred")
        axes[i, 1].set_xticks(x)
        axes[i, 1].set_xticklabels(labels, rotation=90)
        axes[i, 1].set_ylim(0, 1.05)
        axes[i, 1].set_title("Concept Activations")
        axes[i, 1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
