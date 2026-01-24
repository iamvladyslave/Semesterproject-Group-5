from __future__ import annotations

import argparse
from pathlib import Path

try:
    import scripts.train_cbm as train_cbm
    import scripts.evaluate_cbm as evaluate_cbm
except ModuleNotFoundError:  # Allows running via python scripts/run.py
    import train_cbm  # type: ignore
    import evaluate_cbm  # type: ignore


def _train(args) -> None:
    concept_epochs = args.epoch if args.epoch is not None else args.concept_epochs
    label_epochs = args.epoch if args.epoch is not None else args.label_epochs
    train_args = argparse.Namespace(
        data_config=args.data_config,
        training_config=args.training_config,
        backbone=args.backbone,
        no_pretrained=args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        threshold=args.threshold,
        binary_concepts=args.binary_concepts,
        concept_epochs=concept_epochs,
        label_epochs=label_epochs,
        save_dir=args.save_dir,
        num_examples=args.num_examples,
        max_concepts=args.max_concepts,
    )
    train_cbm.main(train_args)


def _eval(args, concept_ckpt: Path, label_ckpt: Path) -> None:
    eval_args = argparse.Namespace(
        data_config=args.data_config,
        concept_ckpt=str(concept_ckpt),
        label_ckpt=str(label_ckpt),
        backbone=args.backbone,
        dropout=args.dropout,
        threshold=args.threshold,
        binary_concepts=args.binary_concepts,
        split=args.split,
        save_dir=args.eval_dir,
        predictions_csv=args.predictions_csv,
        num_examples=args.num_examples,
        max_concepts=args.max_concepts,
    )
    evaluate_cbm.main(eval_args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plug-and-play runner for CBM training + evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "all"),
        default="all",
        help="Run training, evaluation, or both.",
    )
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--training-config", default="config/training_config.yaml")
    parser.add_argument("--save-dir", default="artifacts/cbm")
    parser.add_argument("--eval-dir", default="artifacts/cbm_eval")
    parser.add_argument("--concept-ckpt", default=None)
    parser.add_argument("--label-ckpt", default=None)
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--concept-epochs", type=int, default=None)
    parser.add_argument("--label-epochs", type=int, default=None)
    parser.add_argument("--backbone", default="efficientnet_v2_s")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--binary-concepts", dest="binary_concepts", action="store_true")
    parser.add_argument("--no-binary-concepts", dest="binary_concepts", action="store_false")
    parser.set_defaults(binary_concepts=True)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--max-concepts", type=int, default=None)

    args = parser.parse_args()
    if args.mode in ("train", "all"):
        _train(args)

    if args.mode in ("eval", "all"):
        concept_ckpt = (
            Path(args.concept_ckpt)
            if args.concept_ckpt
            else Path(args.save_dir) / "concept_predictor.pt"
        )
        label_ckpt = (
            Path(args.label_ckpt)
            if args.label_ckpt
            else Path(args.save_dir) / "label_predictor.pt"
        )
        if not concept_ckpt.exists():
            raise FileNotFoundError(f"Missing concept checkpoint: {concept_ckpt}")
        if not label_ckpt.exists():
            raise FileNotFoundError(f"Missing label checkpoint: {label_ckpt}")
        _eval(args, concept_ckpt, label_ckpt)


if __name__ == "__main__":
    main()
