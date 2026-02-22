from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.transforms import build_transforms
from src.utils.paths import resolve_project_path


def _resolve_split_paths(data_cfg: dict[str, Any]) -> dict[str, Path]:
    root = resolve_project_path(data_cfg["root_dir"])
    split_paths = {
        "train": root / data_cfg.get("train_dir", "train"),
        "valid": root / data_cfg.get("valid_dir", "valid"),
        "test": root / data_cfg.get("test_dir", "test"),
    }
    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            raise FileNotFoundError(
                f"Expected split folder not found for '{split_name}': {split_path}"
            )
    return split_paths


def build_datasets(data_cfg: dict[str, Any]) -> dict[str, ImageFolder]:
    split_paths = _resolve_split_paths(data_cfg)
    tfms = build_transforms()
    datasets = {
        split: ImageFolder(root=str(path), transform=tfms[split])
        for split, path in split_paths.items()
    }
    return datasets


def build_dataloaders(
    datasets: dict[str, ImageFolder], dataloader_cfg: dict[str, Any]
) -> dict[str, DataLoader]:
    batch_size = int(dataloader_cfg.get("batch_size", 32))
    num_workers = int(dataloader_cfg.get("num_workers", 2))
    pin_memory = bool(dataloader_cfg.get("pin_memory", True))

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def compute_class_weights_from_targets(
    targets: list[int], device: torch.device
) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(targets, dtype=torch.long)).float()
    if torch.any(counts == 0):
        counts = counts + (counts == 0).float()
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return weights.to(device)
