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
    plot_hamming_histogram,
    plot_score_bars,
)
from src.models import ConceptBackboneConfig, ConceptPredictor, LabelPredictor


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _unwrap_dataset(ds):
    return ds.dataset if isinstance(ds, Subset) else ds


def _has_labels(ds) -> bool:
    base = _unwrap_dataset(ds)
    return getattr(base, "has_labels", True)


def _filenames_for_dataset(ds) -> list[str]:
    base = _unwrap_dataset(ds)
    if not hasattr(base, "samples"):
        raise AttributeError("Dataset does not expose samples for filename export.")
    if isinstance(ds, Subset):
        indices = ds.indices
    else:
        indices = range(len(base.samples))
    return [Path(base.samples[i][0]).name for i in indices]


def _predict_labels(concept_model, label_model, dataloader, device, threshold, binary_concepts):
    concept_model.eval()
    label_model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            concept_logits = concept_model(images)
            concept_probs = torch.sigmoid(concept_logits)
            if binary_concepts:
                concept_inputs = (concept_probs >= threshold).float()
            else:
                concept_inputs = concept_probs
            label_logits = label_model(concept_inputs)
            batch_preds = label_logits.argmax(dim=1).cpu().tolist()
            preds.extend(batch_preds)
    return preds


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
    concept_preds = (torch.sigmoid(concept_logits) >= threshold).int()
    hamming_distances = (concept_preds != concept_targets).sum(dim=1).cpu().tolist()

    examples = None
    if example_images:
        examples = {
            "images": torch.cat(example_images, dim=0),
            "true_labels": example_true,
            "pred_labels": example_pred,
            "concept_targets": torch.cat(example_concept_targets, dim=0),
            "concept_preds": torch.cat(example_concept_preds, dim=0),
        }
    return concept_eval, label_eval, hamming_distances, examples


def main(args):
    data_cfg = load_yaml(args.data_config)
    _, _, test_ds, _, _, test_loader = build_dataloaders(data_cfg)
    if test_loader is None or test_ds is None:
        raise RuntimeError("test loader is required for evaluation.")

    base_ds = _unwrap_dataset(test_ds)
    num_concepts = base_ds.num_concepts
    num_classes = len(base_ds.classes)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but was not detected.")
    device = torch.device("cuda")
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

    if not _has_labels(test_ds):
        preds = _predict_labels(
            concept_model,
            label_model,
            test_loader,
            device,
            threshold=args.threshold,
            binary_concepts=args.binary_concepts,
        )
        filenames = _filenames_for_dataset(test_ds)
        if len(preds) != len(filenames):
            raise RuntimeError("Prediction count does not match test filenames.")

        out_dir = Path(args.save_dir) if args.save_dir else Path("artifacts/cbm_eval")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.predictions_csv) if args.predictions_csv else out_dir / "test_predictions.csv"
        with open(out_path, "w") as f:
            f.write("Filename,ClassId\n")
            for name, pred in zip(filenames, preds):
                f.write(f"{name},{pred}\n")
        print(f"Unlabeled test set detected. Predictions saved to {out_path}")
        return

    concept_eval, label_eval, hamming_distances, examples = evaluate_cbm(
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
        plot_confusion_matrix(label_eval["confusion_matrix"], save_path=out_dir / "confusion_matrix.png")
        plot_score_bars(
            concept_eval["per_concept_f1"],
            labels=getattr(base_ds, "concept_columns", None),
            title="Per-Concept F1 (Worst to Best)",
            ylabel="F1",
            sort_asc=True,
            max_items=10,
            save_path=out_dir / "per_concept_f1.png",
        )
        plot_score_bars(
            label_eval["per_class_f1"],
            labels=getattr(base_ds, "classes", None),
            title="Per-Class F1 (Worst to Best)",
            ylabel="F1",
            sort_asc=True,
            max_items=10,
            save_path=out_dir / "per_class_f1.png",
        )
        avg_hamming = sum(hamming_distances) / len(hamming_distances) if hamming_distances else 0.0
        exact_match = concept_eval.get("exact_match", 0.0)
        plot_hamming_histogram(
            hamming_distances,
            title=f"Hamming Distance (avg={avg_hamming:.2f}, exact={exact_match:.3f})",
            save_path=out_dir / "hamming_distance.png",
        )

        if examples is not None:
            plot_example_predictions(
                examples["images"],
                true_labels=examples["true_labels"],
                pred_labels=examples["pred_labels"],
                concept_targets=examples["concept_targets"],
                concept_preds=examples["concept_preds"],
                concept_names=base_ds.concept_columns,
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
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--binary-concepts", dest="binary_concepts", action="store_true")
    parser.add_argument("--no-binary-concepts", dest="binary_concepts", action="store_false")
    parser.set_defaults(binary_concepts=True)
    parser.add_argument("--save-dir", type=str, default="artifacts/cbm_eval")
    parser.add_argument("--predictions-csv", type=str, default=None)
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--max-concepts", type=int, default=None)
    args = parser.parse_args()
    main(args)
