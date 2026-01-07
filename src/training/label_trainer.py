import torch
import torch.nn as nn


class LabelTrainer:
    def __init__(
        self,
        concept_predictor,
        label_predictor,
        train_loader,
        val_loader,
        optimizer,
        device,
        threshold=0.5,
        binary_concepts=True,
        loss_fn=None,
    ):
        self.concept_predictor = concept_predictor.to(device)
        self.label_predictor = label_predictor.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.threshold = threshold
        self.binary_concepts = binary_concepts
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        #freeze concept model
        self.concept_predictor.eval()
        for p in self.concept_predictor.parameters():
            p.requires_grad = False

    def _concepts(self, images):
        with torch.no_grad():
            logits = self.concept_predictor(images)
            probs = torch.sigmoid(logits)
            if self.binary_concepts:
                probs = (probs >= self.threshold).float()
            return probs

    def train_epoch(self, dataloader, train=True):
        if train:
            self.label_predictor.train()
        else:
            self.label_predictor.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, (_, labels) in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            concepts = self._concepts(images)
            logits = self.label_predictor(concepts)
            loss = self.loss_fn(logits, labels)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        acc = total_correct / total_samples if total_samples else 0.0
        return avg_loss, acc

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(self.train_loader, train=True)
            val_loss, val_acc = self.train_epoch(self.val_loader, train=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        return self.history
