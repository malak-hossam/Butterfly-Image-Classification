from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    preds = torch.argmax(logits, dim=1)
    return int((preds == labels).sum().item())


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    device: torch.device,
    desc: str = "Train",
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, desc=desc, leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += _accuracy(logits, labels)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    desc: str = "Eval",
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, desc=desc, leave=False)
    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += _accuracy(logits, labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy
