from __future__ import annotations

from typing import Dict, List, Optional

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
