# Concept Bottleneck Model - Concept Predictor

This repo now includes a concept prediction stage for GTSRB traffic sign images.

## Train the concept predictor
```
python -m scripts.train_concept_predictor \
    --data-config config/data_config.yaml \
    --training-config config/training_config.yaml \
    --save-dir artifacts/concept_predictor
```
Flags:
- `--no-pretrained` disable ImageNet weights for the EfficientNetV2 backbone.
- `--freeze-backbone` freeze backbone parameters (head only trainable).
- `--backbone` choose EfficientNet variant (default `efficientnet_v2_s`).
- `--threshold` set sigmoid cutoff for concept predictions (default 0.5).
- `--epoch` set N amount of epochs (default 30).

Artifacts: model weights, training curves, and per-concept accuracy plots are written to `--save-dir`.

~~Takes about an hour for now, change epoch to 5 or so to get sanity-check results in 10-15 minutes. (config/training_config.yaml)~~
Runs on Gruenau without issues, no full run done yet.

### TODO: (from most to least important)

- currently random split for train/val!!! implement correct train/val/test split!!!
- **conda environment for linux should be default (currently done through semproServer conda env)**; this effectively means mac environment implementation is not useful anymore, MPS/CPU-only if-else case is to be taken out and shot.
- more visualization, more visualization.py
- decide if artifacts should keep plots and visualization in repo (currently ignored gitignore)
- fiddle with patience (at 10 meaning 10 consqutive epochs of no val_loss improvement) and min_delta (lower) values!
- clean clutter and irrelavant directories/files
- ~~update gitignore, remove pycache etc.~~
- start on task 3
- simplify script to run (task 4)
- extend README for complete setup (task 4)
- notebook with snippets for demo for final?
