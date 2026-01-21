# Concept Bottleneck Model - Concept Predictor

## Quick run (train + evaluate)
```
python -m scripts.run
```
only CUDA



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

Artifacts: model weights, training curves, and metrics are written to `--save-dir`.

Current full run takes ca. 40 minutes,
    early stopping on epochs 22 and 14 respectively.

### TODO: (from most to least important)

- extend README for complete setup (task 4)
- notebook with snippets for demo for final?
