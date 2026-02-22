# System Design

## Overview

This repository implements two image-classification training pipelines for butterfly/moth species:

1. Custom CNN (4 convolution blocks from notebook).
2. Transfer learning with ResNet18 (ImageNet-pretrained backbone).

Both pipelines share a common data layer, transform policy, training loop mechanics, checkpointing behavior, and evaluation outputs.

## Data Flow

1. Config loading from YAML (`src/config/*.yaml`) with base-config merge.
2. Seed setup for reproducibility (`src/utils/seed.py`).
3. ImageFolder datasets initialized from:
   - `data/raw/butterfly-images40-species/train`
   - `data/raw/butterfly-images40-species/valid`
   - `data/raw/butterfly-images40-species/test`
4. Pipeline-specific model initialization:
   - `src/models/cnn_model.py`
   - `src/models/resnet18_model.py`
5. Training/evaluation loop with `CrossEntropyLoss`, Adam optimizer, tqdm bars.
6. Best checkpoint saved by validation accuracy.
7. Metrics JSON emitted for experiment traceability.
8. Post-training evaluation produces confusion matrix and classification report.
9. Inference supports single image and recursive folder mode.

## Transform Strategy

Train transforms:

- `RandomResizedCrop(224)`
- `RandomHorizontalFlip()`
- `RandomRotation(15)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`
- `ToTensor()`
- `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`

Validation/test transforms:

- `Resize(256)`
- `CenterCrop(224)`
- `ToTensor()`
- Same ImageNet normalization

## Model Design

### Custom CNN

- Conv blocks: `3->64->128->256->512`, each with `ReLU + MaxPool2d(2)`.
- Input resolution: `224x224`.
- Flatten output: `512 * 14 * 14`.
- Classifier:
  - `fc1: 512*14*14 -> 1024`
  - `dropout(0.5)`
  - `fc2: 1024 -> 512`
  - `fc3: 512 -> num_classes`

### ResNet18 Transfer

- `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` when enabled.
- Final FC replaced to match inferred class count.
- Optional backbone freezing (`freeze_backbone=true`) keeps only classifier trainable.

## Training Loop and Checkpointing

- Shared functions:
  - `train_one_epoch`
  - `evaluate`
- Tracked per epoch:
  - train loss/accuracy
  - val loss/accuracy
  - test loss/accuracy
- Best checkpoint criterion: max validation accuracy.
- Checkpoint path:
  - `outputs/models/<pipeline_name>/best.pt`
- Metrics path:
  - `outputs/metrics/<pipeline_name>/metrics.json`

## Evaluation and Inference

Evaluation (`src/eval/evaluate.py`):

- Loads checkpoint and selected split.
- Computes accuracy, confusion matrix, classification report.
- Saves confusion matrix image under `outputs/plots/`.
- Saves report JSON under `outputs/metrics/<pipeline_name>/`.

Inference (`src/infer/predict.py`):

- Single image: prints predicted class and confidence.
- Folder mode: recursively scans images and writes `outputs/predictions/preds.csv`.
- Supports `top_k` probabilities.

## MLOps-lite Considerations

- Config-driven execution for reproducible runs.
- Deterministic-ish behavior with fixed seeds.
- Isolated outputs for models/metrics/plots/predictions.
- CI smoke tests for imports and transforms.
- Dockerfile for portable runtime.

## Trade-offs

- Simplicity over framework-heavy abstractions (no Lightning/Hydra).
- Determinism is best-effort; exact bitwise reproducibility across CUDA/cuDNN versions is not guaranteed.
- Confusion matrix with many classes is information-dense and less visually readable than per-class metrics tables.

