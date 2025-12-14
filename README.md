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

### TODO:

- simplify script to run
- conda environment for linux should be default (currently done through semproServer conda env)
- clean clutter and irrelavant directories/files
- ~~update gitignore, remove pycache etc.~~
- start on task 3