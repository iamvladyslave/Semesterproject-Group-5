from __future__ import annotations

import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import Subset

from src.data.splits import build_dataloaders
from src.models import ConceptBackboneConfig, ConceptPredictor
from src.training.concept_trainer import ConceptTrainer, EarlyStopping
from src.evaluation.metrics import concept_metrics
from src.evaluation.visualization import plot_per_concept_performance, plot_training_curves


def load_yaml(path: str | Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _num_concepts(ds):
    if hasattr(ds, "num_concepts"):
        return ds.num_concepts
    if isinstance(ds, Subset):
        return ds.dataset.num_concepts
    raise AttributeError("Dataset does not expose num_concepts")


def main(args):
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.training_config)

    train_ds, val_ds, _, train_loader, val_loader, _ = build_dataloaders(data_cfg)
    if val_loader is None:
        raise RuntimeError("Validation loader is required for concept training.")
    num_concepts = _num_concepts(train_ds)

    backbone_cfg = ConceptBackboneConfig(
        name=args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    )
    model = ConceptPredictor(num_concepts=num_concepts, backbone_cfg=backbone_cfg)

    #only CUDA; fail fast if unavailable
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but was not detected.")
    device = torch.device(train_cfg["training"].get("device", "cuda"))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        weight_decay=train_cfg["training"].get("weight_decay", 0.0),
    )
    early_stopping = EarlyStopping(
        patience=train_cfg["training"].get("patience", 10),
        min_delta=train_cfg["training"].get("min_delta", 0.0),
    )
    trainer = ConceptTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        threshold=args.threshold,
        early_stopping=early_stopping,
    )

    max_epochs = args.epochs if args.epochs is not None else train_cfg["training"]["epochs"]
    history = trainer.fit(max_epochs)
    print("Training finished. Best val loss:", early_stopping.best_loss)

    # Evaluate on validation set
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, (concepts, _) in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(concepts)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = concept_metrics(logits, targets, threshold=args.threshold)
    print("Macro metrics:", metrics["macro"])
    print(f"Exact-match accuracy: {metrics['macro']['exact_match']:.4f}")

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(history.__dict__, save_path=out_dir / "training_curves.png")
        plot_per_concept_performance(
            metrics["per_concept_accuracy"],
            title="Per-Concept Accuracy",
            save_path=out_dir / "per_concept_accuracy.png",
        )
        torch.save(model.state_dict(), out_dir / "concept_predictor.pt")
        with open(out_dir / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics, f)
        print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train concept predictor on GTSRB concepts.")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--training-config", default="config/training_config.yaml")
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="artifacts/concept_predictor")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from training config")
    args = parser.parse_args()
    main(args)
