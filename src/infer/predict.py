from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src.data.datamodule import build_datasets
from src.data.transforms import get_eval_transforms
from src.models.cnn_model import CustomCNN
from src.models.resnet18_model import build_resnet18
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import prepare_output_dirs
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict butterfly/moth classes.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps/auto")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image_path", type=str, default=None, help="Single image path")
    mode.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory for recursive inference",
    )
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
            freeze_backbone=False,
            pretrained=False,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def predict_topk(
    model: nn.Module,
    image_path: Path,
    transform,
    device: torch.device,
    idx_to_class: dict[int, str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)

    preds: list[tuple[str, float]] = []
    for prob, idx in zip(top_probs[0].cpu().tolist(), top_idxs[0].cpu().tolist()):
        preds.append((idx_to_class[idx], float(prob)))
    return preds


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(args.device, str(cfg.get("device", "auto")))
    pipeline_name = cfg["project"]["pipeline_name"]
    outputs = prepare_output_dirs(pipeline_name, cfg["outputs"]["root_dir"])

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx")
    if class_to_idx is None:
        datasets = build_datasets(cfg["data"])
        class_to_idx = datasets["train"].class_to_idx

    num_classes = len(class_to_idx)
    model_name = str(checkpoint.get("model_name", cfg["model"]["name"])).lower()
    model = build_model(model_name, num_classes, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    transform = get_eval_transforms()

    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        preds = predict_topk(
            model=model,
            image_path=image_path,
            transform=transform,
            device=device,
            idx_to_class=idx_to_class,
            top_k=args.top_k,
        )
        top_label, top_prob = preds[0]
        LOGGER.info("Image: %s", image_path)
        LOGGER.info("Predicted: %s (%.4f)", top_label, top_prob)
        LOGGER.info("Top-%d: %s", args.top_k, preds)
        return

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    images = [
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found under: {image_dir}")

    records: list[dict[str, str | float]] = []
    for image_path in images:
        preds = predict_topk(
            model=model,
            image_path=image_path,
            transform=transform,
            device=device,
            idx_to_class=idx_to_class,
            top_k=args.top_k,
        )
        top_label, top_prob = preds[0]
        top_k_repr = "; ".join([f"{label}:{prob:.4f}" for label, prob in preds])
        records.append(
            {
                "image_path": str(image_path),
                "predicted_class": top_label,
                "probability": top_prob,
                "top_k": top_k_repr,
            }
        )

    df = pd.DataFrame(records)
    out_csv = Path(outputs["predictions_dir"]) / "preds.csv"
    df.to_csv(out_csv, index=False)
    LOGGER.info("Processed %d images from %s", len(images), image_dir)
    LOGGER.info("Saved predictions CSV: %s", out_csv)


if __name__ == "__main__":
    main()
