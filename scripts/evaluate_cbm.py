import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data.preprocessing import Preprocessing
from src.dataset import GTSRBConceptDataset
from src.evaluation.metrics import concept_metrics, label_metrics
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_example_predictions,
    plot_per_concept_performance,
)
from src.models import ConceptBackboneConfig, ConceptPredictor, LabelPredictor


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg):
    dataset_cfg = cfg["dataset"]
    dataloader_cfg = cfg["dataloader"]

    image_size = dataset_cfg["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    preprocessor = Preprocessing(image_size=image_size)
    tf = preprocessor.transform
    ds = GTSRBConceptDataset(
        root_dir=dataset_cfg["root_dir"],
        concepts_csv=dataset_cfg["concepts_csv"],
        transform=tf,
        image_exts=tuple(dataset_cfg.get("image_exts", [".ppm"])),
        crop_with_roi=True,
    )

    val_split = dataset_cfg.get("val_split", 0.1)
    test_split = dataset_cfg.get("test_split", 0.1)
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    labels = [lbl for _, lbl in ds.samples]
    indices = list(range(len(labels)))

    if val_split > 0:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            random_state=dataset_cfg.get("seed", 1923),
            stratify=labels,
        )
    else:
        train_idx, val_idx = indices, []

    test_idx = []
    if test_split > 0:
        train_labels = [labels[i] for i in train_idx]
        train_idx, test_idx = train_test_split(
            train_idx,
            test_size=test_split / max(1e-8, (1 - val_split)),
            random_state=dataset_cfg.get("seed", 1923),
            stratify=train_labels,
        )

    test_ds = Subset(ds, test_idx) if test_idx else None

    def seed_worker(worker_id: int):
        worker_seed = dataset_cfg.get("seed", 1923) + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    generator = torch.Generator().manual_seed(dataset_cfg.get("seed", 1923))

    def make_loader(split_ds):
        if split_ds is None:
            return None
        return DataLoader(
            split_ds,
            batch_size=dataloader_cfg["batch_size"],
            shuffle=False,
            num_workers=dataloader_cfg["num_workers"],
            pin_memory=dataloader_cfg.get("pin_memory", True),
            worker_init_fn=seed_worker,
            generator=generator,
        )

    return ds, make_loader(test_ds)


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
    ds, test_loader = build_dataloaders(data_cfg)
    if test_loader is None:
        raise RuntimeError("test loader is required for evaluation.")

    num_concepts = ds.num_concepts
    num_classes = len(ds.classes)

    device = torch.device(args.device)
    backbone_cfg = ConceptBackboneConfig(
        name=args.backbone,
        pretrained=False,
        dropout=args.dropout,
        freeze_backbone=False,
    )
    concept_model = ConceptPredictor(num_concepts=num_concepts, backbone_cfg=backbone_cfg)
    label_model = LabelPredictor(num_concepts=num_concepts, num_classes=num_classes)

    concept_model.load_state_dict(torch.load(args.concept_ckpt, map_location=device))
    label_model.load_state_dict(torch.load(args.label_ckpt, map_location=device))
    concept_model.to(device)
    label_model.to(device)

    concept_eval, label_eval, examples = evaluate_cbm(
        concept_model,
        label_model,
        test_loader,
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
        plot_per_concept_performance(
            concept_eval["per_concept_accuracy"],
            title="Per-Concept Accuracy (Test)",
            save_path=out_dir / "per_concept_accuracy_test.png",
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

        with open(out_dir / "concept_metrics.yaml", "w") as f:
            yaml.safe_dump(concept_eval, f)
        with open(out_dir / "label_metrics.yaml", "w") as f:
            yaml.safe_dump(label_eval, f)
        print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CBM models on test split.")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--concept-ckpt", default="artifacts/cbm/concept_predictor.pt")
    parser.add_argument("--label-ckpt", default="artifacts/cbm/label_predictor.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--binary-concepts", dest="binary_concepts", action="store_true")
    parser.add_argument("--no-binary-concepts", dest="binary_concepts", action="store_false")
    parser.set_defaults(binary_concepts=True)
    parser.add_argument("--save-dir", type=str, default="artifacts/cbm_eval")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--max-concepts", type=int, default=None)
    args = parser.parse_args()
    main(args)
