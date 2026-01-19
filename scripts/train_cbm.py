import argparse
from pathlib import Path
import torch
import yaml
from torch.utils.data import Subset

from src.data.splits import build_dataloaders
from src.evaluation.metrics import concept_metrics, label_metrics
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_example_predictions,
    plot_per_concept_performance,
    plot_training_curves,
)
from src.models import ConceptBackboneConfig, ConceptPredictor, LabelPredictor
from src.training.concept_trainer import ConceptTrainer, EarlyStopping
from src.training.label_trainer import LabelTrainer


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _unwrap_dataset(ds):
    return ds.dataset if isinstance(ds, Subset) else ds


def evaluate_cbm(concept_model, label_model, dataloader, device, threshold, binary_concepts, max_examples):
    concept_model.eval()
    label_model.eval()

    all_concept_logits = []
    all_concept_targets = []
    all_label_logits = []
    all_labels = []

    example_images = []
    example_true = []
    example_pred = []
    example_concept_targets = []
    example_concept_preds = []
    examples_collected = 0

    with torch.no_grad():
        for images, (concept_targets, labels) in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            concept_targets = concept_targets.to(device)

            concept_logits = concept_model(images)
            concept_probs = torch.sigmoid(concept_logits)
            if binary_concepts:
                concept_inputs = (concept_probs >= threshold).float()
            else:
                concept_inputs = concept_probs
            label_logits = label_model(concept_inputs)

            all_concept_logits.append(concept_logits.cpu())
            all_concept_targets.append(concept_targets.cpu())
            all_label_logits.append(label_logits.cpu())
            all_labels.append(labels.cpu())

            if examples_collected < max_examples:
                remaining = max_examples - examples_collected
                take = min(remaining, images.size(0))
                preds = label_logits.argmax(dim=1)
                example_images.append(images[:take].cpu())
                example_true.extend(labels[:take].cpu().tolist())
                example_pred.extend(preds[:take].cpu().tolist())
                example_concept_targets.append(concept_targets[:take].cpu())
                example_concept_preds.append(concept_inputs[:take].cpu())
                examples_collected += take

    concept_logits = torch.cat(all_concept_logits, dim=0)
    concept_targets = torch.cat(all_concept_targets, dim=0)
    label_logits = torch.cat(all_label_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    concept_eval = concept_metrics(concept_logits, concept_targets, threshold=threshold)
    label_eval = label_metrics(label_logits, labels, num_classes=label_logits.shape[1])

    examples = None
    if example_images:
        examples = {
            "images": torch.cat(example_images, dim=0),
            "true_labels": example_true,
            "pred_labels": example_pred,
            "concept_targets": torch.cat(example_concept_targets, dim=0),
            "concept_preds": torch.cat(example_concept_preds, dim=0),
        }
    return concept_eval, label_eval, examples


def main(args):
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.training_config)

    train_ds, val_ds, _, train_loader, val_loader, _ = build_dataloaders(data_cfg)
    if val_loader is None:
        raise RuntimeError("Validation loader is required for CBM training.")

    base_ds = _unwrap_dataset(train_ds)
    num_concepts = base_ds.num_concepts
    num_classes = len(base_ds.classes)

    device_name = train_cfg["training"].get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but was not detected.")
    device = torch.device(device_name)

    backbone_cfg = ConceptBackboneConfig(
        name=args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    )
    concept_model = ConceptPredictor(num_concepts=num_concepts, backbone_cfg=backbone_cfg)

    optimizer = torch.optim.Adam(
        concept_model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        weight_decay=train_cfg["training"].get("weight_decay", 0.0),
    )
    early_stopping = EarlyStopping(
        patience=train_cfg["training"].get("patience", 10),
        min_delta=train_cfg["training"].get("min_delta", 0.0),
    )
    concept_trainer = ConceptTrainer(
        model=concept_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        threshold=args.threshold,
        early_stopping=early_stopping,
    )
    concept_epochs = args.concept_epochs or train_cfg["training"]["epochs"]
    concept_history = concept_trainer.fit(concept_epochs)

    label_model = LabelPredictor(num_concepts=num_concepts, num_classes=num_classes)
    label_optimizer = torch.optim.Adam(
        label_model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        weight_decay=train_cfg["training"].get("weight_decay", 0.0),
    )
    label_early_stopping = EarlyStopping(
        patience=train_cfg["training"].get("patience", 10),
        min_delta=train_cfg["training"].get("min_delta", 0.0),
    )
    label_trainer = LabelTrainer(
        concept_predictor=concept_model,
        label_predictor=label_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=label_optimizer,
        device=device,
        threshold=args.threshold,
        binary_concepts=args.binary_concepts,
        early_stopping=label_early_stopping,
    )
    label_epochs = args.label_epochs or train_cfg["training"]["epochs"]
    label_history = label_trainer.fit(label_epochs)

    concept_eval, label_eval, examples = evaluate_cbm(
        concept_model,
        label_model,
        val_loader,
        device,
        threshold=args.threshold,
        binary_concepts=args.binary_concepts,
        max_examples=args.num_examples,
    )
    print("Concept macro metrics:", concept_eval["macro"])
    print("Label accuracy:", label_eval["accuracy"])

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_training_curves(concept_history.__dict__, save_path=out_dir / "concept_training_curves.png")
        plot_training_curves(label_history, save_path=out_dir / "label_training_curves.png")
        plot_per_concept_performance(
            concept_eval["per_concept_accuracy"],
            title="Per-Concept Accuracy (Val)",
            save_path=out_dir / "per_concept_accuracy_val.png",
        )
        plot_confusion_matrix(label_eval["confusion_matrix"], save_path=out_dir / "confusion_matrix.png")

        if examples is not None:
            plot_example_predictions(
                examples["images"],
                true_labels=examples["true_labels"],
                pred_labels=examples["pred_labels"],
                concept_targets=examples["concept_targets"],
                concept_preds=examples["concept_preds"],
                concept_names=ds.concept_columns,
                max_concepts=args.max_concepts,
                save_path=out_dir / "example_predictions.png",
            )

        torch.save(concept_model.state_dict(), out_dir / "concept_predictor.pt")
        torch.save(label_model.state_dict(), out_dir / "label_predictor.pt")
        with open(out_dir / "concept_metrics.yaml", "w") as f:
            yaml.safe_dump(concept_eval, f)
        with open(out_dir / "label_metrics.yaml", "w") as f:
            yaml.safe_dump(label_eval, f)
        print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequentially train concept and label predictors.")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--training-config", default="config/training_config.yaml")
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--binary-concepts", dest="binary_concepts", action="store_true")
    parser.add_argument("--no-binary-concepts", dest="binary_concepts", action="store_false")
    parser.set_defaults(binary_concepts=True)
    parser.add_argument("--concept-epochs", type=int, default=None)
    parser.add_argument("--label-epochs", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="artifacts/cbm")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--max-concepts", type=int, default=None)
    args = parser.parse_args()
    main(args)
