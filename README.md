# Concept Bottleneck Model GTSRB

## Initial Setup

### Use Conda to set up environment after git pull repo:

0. meant for gruenau servers only, mostly run on gruenau1
1. conda env create --file=environment.yaml
(GTSRB Dataset is included in the Repo, no need to seperately download, make sure data_config.yaml points to correct directory if download was still done seperately)

2. change into directory of project
3. conda env activate sempro
4. run script based on needs

## Quick run (train + evaluate)

```
python -m scripts.run
```
only CUDA

## Evaluate trained CBM
```
python -m scripts.evaluate_cbm \
    --data-config config/data_config.yaml \
    --concept-ckpt artifacts/cbm/concept_predictor.pt \
    --label-ckpt artifacts/cbm/label_predictor.pt \
    --save-dir artifacts/cbm_eval
```
Flags:
- `--split` choose `test` or `val` (default `test`).
- `--predictions-csv` write CSV for unlabeled split predictions.
- `--threshold` set sigmoid cutoff for concept predictions (default 0.5).

## Train concept predictor
```
python -m scripts.train_concept_predictor \
    --data-config config/data_config.yaml \
    --training-config config/training_config.yaml \
    --save-dir artifacts/concept_predictor
```

Trains only concept predictor (no label head). Full CBM training + eval done by `scripts.run`
Flags:
- `--no-pretrained` disable ImageNet weights for the EfficientNetV2 backbone.
- `--freeze-backbone` freeze backbone parameters (head only trainable).
- `--backbone` choose EfficientNet variant (default `efficientnet_v2_s`).
- `--dropout` set classifier dropout (default 0.2).
- `--threshold` set sigmoid cutoff for concept predictions (default 0.5).
- `--epoch` override epochs from training config (default 30).

Artifacts: model weights, training curves, and metrics are written to `--save-dir`

Current full run takes ca. 40 minutes; early stopping on epochs 22 and 14 respectively.

### TODO: (from most to least important)

- extend README for complete setup (task 4)
- notebook with snippets for demo for final?
