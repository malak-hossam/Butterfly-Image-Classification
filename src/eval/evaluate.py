from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.data.datamodule import (
    build_dataloaders,
    build_datasets,
    compute_class_weights_from_targets,
)
from src.models.cnn_model import CustomCNN
from src.models.metrics import evaluate
from src.models.resnet18_model import build_resnet18
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import prepare_output_dirs
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to outputs/models/<pipeline>/best.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "valid", "test"],
        help="Dataset split for evaluation.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps/auto")
    return parser.parse_args()


def resolve_device(device_arg: str | None, cfg_device: str) -> torch.device:
    choice = device_arg or cfg_device
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def build_model(model_name: str, num_classes: int, cfg: dict) -> nn.Module:
    if model_name == "cnn":
        return CustomCNN(
            num_classes=num_classes,
            dropout=float(cfg["model"].get("dropout", 0.5)),
        )
    if model_name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            freeze_backbone=bool(cfg["training"].get("freeze_backbone", False)),
            pretrained=False,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def save_confusion_matrix(cm: list[list[int]] | torch.Tensor, path: Path, class_names: list[str]) -> None:
    cm_tensor = torch.tensor(cm)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_tensor.numpy(), cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if len(class_names) <= 30:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    device = resolve_device(args.device, str(cfg.get("device", "auto")))
    split = args.split or cfg["evaluation"].get("split", "test")
    pipeline_name = cfg["project"]["pipeline_name"]
    outputs = prepare_output_dirs(pipeline_name, cfg["outputs"]["root_dir"])

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(outputs["model_dir"]) / "best.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    datasets = build_datasets(cfg["data"])
    loaders = build_dataloaders(datasets, cfg["dataloader"])
    class_names = datasets["train"].classes
    num_classes = len(class_names)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_name = str(checkpoint.get("model_name", cfg["model"]["name"])).lower()
    model = build_model(model_name, num_classes, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    LOGGER.info("Loaded checkpoint: %s", ckpt_path)

    class_weights = None
    if bool(cfg["training"].get("use_class_weights", False)):
        class_weights = compute_class_weights_from_targets(datasets["train"].targets, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    loss, accuracy = evaluate(
        model=model,
        dataloader=loaders[split],
        criterion=criterion,
        device=device,
        desc=f"Evaluate {split}",
    )

    all_targets: list[int] = []
    all_preds: list[int] = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loaders[split], desc="Collect predictions", leave=False):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.tolist())

    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm_path = Path(outputs["plots_dir"]) / f"{pipeline_name}_{split}_confusion_matrix.png"
    save_confusion_matrix(cm.tolist(), cm_path, class_names)

    eval_payload = {
        "pipeline": pipeline_name,
        "model_name": model_name,
        "checkpoint": str(ckpt_path),
        "split": split,
        "loss": loss,
        "accuracy": accuracy,
        "classification_report": report,
    }
    eval_path = Path(outputs["metrics_dir"]) / f"evaluation_{split}.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    LOGGER.info("Evaluation split: %s", split)
    LOGGER.info("Loss: %.4f | Accuracy: %.4f", loss, accuracy)
    LOGGER.info("Saved confusion matrix: %s", cm_path)
    LOGGER.info("Saved evaluation report: %s", eval_path)


if __name__ == "__main__":
    main()
