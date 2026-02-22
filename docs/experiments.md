# Experiments

Use this file as a run log for reproducible experiment tracking.

## Notebook Reference (Baseline from Provided Notebook)

| Pipeline | Epochs | LR | Test Accuracy | Note |
|---|---:|---:|---:|---|
| Custom CNN | 10 | 1e-4 | ~0.726 | Notebook reference |
| ResNet18 transfer | 10 | 1e-3 | ~0.930 | Notebook reference |

These values are reference-only and should be reproduced with script-based runs in this repository.

## Reproduction Log Template

| Date | Pipeline | Config | Seed | Epochs | Val Acc (Best) | Test Acc (Final) | Checkpoint | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| YYYY-MM-DD | cnn / resnet18 | `src/config/*.yaml` | 42 | 10 | - | - | `outputs/models/.../best.pt` | - |

## Suggested Experiment Matrix

1. CNN with and without class weights.
2. ResNet18 with and without frozen backbone.
3. Learning rate sweep (`1e-4`, `3e-4`, `1e-3`).
4. Batch size sweep (`16`, `32`, `64`) based on GPU memory.

## Notes

- Record exact config file and Git commit for each run.
- Save metrics JSON artifacts and confusion matrix plots for every reported result.

