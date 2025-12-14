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

Artifacts: model weights, training curves, and per-concept accuracy plots are written to `--save-dir`.

Takes about an hour for now, change epoch to 5 or so to get sanity-check results in 10-15 minutes. (config/training_config.yaml)
