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

    train_vec_acc = history.get("train_vec_acc", [])
    val_vec_acc = history.get("val_vec_acc", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    if train_vec_acc or val_vec_acc:
        ax[1].plot(train_vec_acc, label="Train (exact-match)")
        ax[1].plot(val_vec_acc, label="Val (exact-match)")
        ax[1].set_title("Exact-Match Accuracy")
    else:
        ax[1].plot(train_acc, label="Train")
        ax[1].plot(val_acc, label="Val")
        ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
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

    # Use float formatting since cm is cast to float for normalization.
    fmt = ".2f" if normalize else ".0f"
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


def plot_score_bars(
    scores: Sequence[float],
    *,
    labels: Optional[Sequence[str]] = None,
    title: str = "Scores",
    ylabel: str = "Score",
    sort_asc: bool = False,
    max_items: Optional[int] = None,
    save_path: Optional[str] = None,
):
    scores_np = np.asarray(scores, dtype=float)
    n = scores_np.shape[0]
    if n == 0:
        return None

    order = np.argsort(scores_np)
    if not sort_asc:
        order = order[::-1]

    if max_items is not None and max_items > 0:
        order = order[: min(max_items, n)]

    labels_list = list(labels) if labels is not None else [str(i) for i in range(n)]
    labels_ordered = [labels_list[i] for i in order]
    scores_ordered = scores_np[order]

    fig_h = max(4, 0.3 * len(scores_ordered))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_pos = np.arange(len(scores_ordered))
    ax.barh(y_pos, scores_ordered)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_ordered)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_hamming_histogram(
    distances: Sequence[int],
    *,
    title: str = "Hamming Distance",
    save_path: Optional[str] = None,
):
    # distances_np = np.asarray(distances, dtype=int)
    # if distances_np.size == 0:
    #     return None
    #
    # max_dist = int(distances_np.max())
    # bins = np.arange(max_dist + 2) - 0.5
    # fig, ax = plt.subplots(figsize=(8, 4))
    # ax.hist(distances_np, bins=bins, edgecolor="black")
    # ax.set_xticks(range(max_dist + 1))
    # ax.set_xlabel("Number of concept errors")
    # ax.set_ylabel("Count")
    # ax.set_title(title)
    #
    # plt.tight_layout()
    # if save_path:
    #     fig.savefig(save_path, bbox_inches="tight")
    # return fig
    return None
