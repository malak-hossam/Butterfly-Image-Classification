from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from src.data.datamodule import (
    build_dataloaders,
    build_datasets,
    compute_class_weights_from_targets,
)
from src.models.cnn_model import CustomCNN
from src.models.metrics import evaluate, train_one_epoch
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import prepare_output_dirs
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom CNN model.")
    parser.add_argument("--config", type=str, default="src/config/cnn.yaml")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps/auto")
    return parser.parse_args()


def resolve_device(device_arg: str | None, cfg_device: str) -> torch.device:
    choice = device_arg or cfg_device
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)

    set_seed(int(cfg["seed"]))
    device = resolve_device(args.device, str(cfg.get("device", "auto")))
    LOGGER.info("Using device: %s", device)

    datasets = build_datasets(cfg["data"])
    loaders = build_dataloaders(datasets, cfg["dataloader"])
    train_ds = datasets["train"]
    num_classes = len(train_ds.classes)
    LOGGER.info("Detected num_classes=%d", num_classes)

    model = CustomCNN(
        num_classes=num_classes,
        dropout=float(cfg["model"].get("dropout", 0.5)),
    ).to(device)

    class_weights = None
    if bool(cfg["training"].get("use_class_weights", False)):
        class_weights = compute_class_weights_from_targets(train_ds.targets, device)
        LOGGER.info("Class weights enabled for loss.")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    pipeline_name = cfg["project"]["pipeline_name"]
    outputs = prepare_output_dirs(pipeline_name, cfg["outputs"]["root_dir"])
    best_ckpt_path = Path(outputs["model_dir"]) / "best.pt"
    metrics_path = Path(outputs["metrics_dir"]) / "metrics.json"

    num_epochs = int(cfg["training"]["epochs"])
    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            desc=f"Train Epoch {epoch}/{num_epochs}",
        )
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=loaders["valid"],
            criterion=criterion,
            device=device,
            desc=f"Valid Epoch {epoch}/{num_epochs}",
        )
        test_loss, test_acc = evaluate(
            model=model,
            dataloader=loaders["test"],
            criterion=criterion,
            device=device,
            desc=f"Test Epoch {epoch}/{num_epochs}",
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        history.append(metrics)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f | test_loss=%.4f test_acc=%.4f",
            epoch,
            num_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            test_loss,
            test_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "model_name": "cnn",
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "classes": train_ds.classes,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "config": cfg,
            }
            torch.save(checkpoint, best_ckpt_path)
            LOGGER.info("Saved new best checkpoint: %s", best_ckpt_path)

    payload = {
        "pipeline": pipeline_name,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "history": history,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Saved metrics JSON: %s", metrics_path)


if __name__ == "__main__":
    main()
