# Model Card

## Models

This repository provides two classification models for butterfly/moth species:

1. Custom CNN (`src/models/cnn_model.py`)
2. ResNet18 transfer model (`src/models/resnet18_model.py`)

## Intended Use

- Educational and applied ML workflows for image classification.
- Benchmarking custom CNN vs transfer learning.
- Baseline for future improvements (augmentation policy, optimizer/scheduler tuning, model scaling).

## Out-of-Scope Use

- Safety-critical or regulatory workflows.
- Species identification for scientific decisions without expert confirmation.

## Training Data

- Kaggle butterfly/moth folder dataset.
- Class count inferred from local training directory.

## Training Procedure

- Loss: CrossEntropyLoss
- Optimizer: Adam
- Evaluation: validation and test splits each epoch
- Checkpointing: best validation accuracy

Optional class weighting:

- `w = 1 / bincount(train_targets)`, then normalized.
- Enabled via config (`training.use_class_weights`).

## Performance (Notebook Reference)

- Custom CNN test accuracy: approximately `0.726`
- ResNet18 test accuracy: approximately `0.930`

These are reference values from the original notebook and should be revalidated locally through reproducible script runs.

## Risks and Limitations

- Domain shift (camera, location, species variants) can degrade performance.
- Dataset bias and long-tail classes may reduce minority-class recall.
- CNN baseline is significantly weaker than transfer learning in reference runs.

## Monitoring Recommendations

- Track per-class precision/recall/F1 from classification report.
- Monitor confusion matrix drift across retrains.
- Log dataset version/hash and config snapshot for each experiment.

