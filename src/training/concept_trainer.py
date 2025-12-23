from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    best_loss: float = float("inf")
    counter: int = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update tracker; returns True when training should stop."""
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter > self.patience


@dataclass
class ConceptTrainerHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_vec_acc: List[float] = field(default_factory=list)
    val_vec_acc: List[float] = field(default_factory=list)
    per_concept_acc: List[List[float]] = field(default_factory=list)


class ConceptTrainer:
    """
    Trainer for the concept predictor head.

    Tracks loss and accuracy (per-concept and macro), supports early stopping,
    and returns the best-performing model weights.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        loss_fn: Optional[nn.Module] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        threshold: float = 0.5,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.threshold = threshold
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
        self.early_stopping = early_stopping or EarlyStopping()
        self.history = ConceptTrainerHistory()

    def _run_epoch(
        self,
        dataloader: DataLoader,
        *,
        train: bool,
    ) -> Tuple[float, float, float, List[float]]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        concept_correct = None
        total_samples = 0
        vector_correct = 0

        for batch in dataloader:
            images, (concepts, _) = batch
            images = images.to(self.device)
            concepts = concepts.to(self.device)

            with torch.set_grad_enabled(train):
                logits = self.model(images)
                loss = self.loss_fn(logits, concepts)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= self.threshold).float()
                correct = (preds == concepts).sum(dim=0)  # per concept
                concept_correct = correct if concept_correct is None else concept_correct + correct
                vector_correct += (preds == concepts).all(dim=1).sum().item()
                total_samples += concepts.shape[0]
                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        per_concept_acc = (concept_correct / total_samples).tolist() if concept_correct is not None else []
        macro_acc = float(sum(per_concept_acc) / len(per_concept_acc)) if per_concept_acc else 0.0
        vector_acc = float(vector_correct / total_samples) if total_samples > 0 else 0.0
        return avg_loss, macro_acc, vector_acc, per_concept_acc

    def fit(self, epochs: int) -> ConceptTrainerHistory:
        for epoch in range(epochs):
            train_loss, train_acc, train_vec_acc, _ = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc, val_vec_acc, per_concept = self._run_epoch(self.val_loader, train=False)

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_acc.append(train_acc)
            self.history.val_acc.append(val_acc)
            self.history.train_vec_acc.append(train_vec_acc)
            self.history.val_vec_acc.append(val_vec_acc)
            self.history.per_concept_acc.append(per_concept)

            if self.scheduler is not None:
                self.scheduler.step()

            stop = self.early_stopping.step(val_loss, self.model)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"train_vec_acc={train_vec_acc:.4f} val_vec_acc={val_vec_acc:.4f}"
            )
            if stop:
                print(
                    f"Early stopping triggered after {self.early_stopping.counter} "
                    f"non-improving epoch(s)."
                )
                break

        if self.early_stopping.best_state is not None:
            self.model.load_state_dict(self.early_stopping.best_state)
        return self.history

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Tuple[float, float, List[float]]:
        dataloader = dataloader or self.val_loader
        loss, macro_acc, vector_acc, per_concept = self._run_epoch(dataloader, train=False)
        return loss, vector_acc, per_concept
