from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch
import torchvision
import matplotlib.pyplot as plt

from src.data.datamodule import build_dataloaders, build_datasets
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import prepare_output_dirs

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageFolder sanity check utility.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        help="Save one sample batch grid into outputs/plots/sample_batch.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)

    datasets = build_datasets(cfg["data"])
    loaders = build_dataloaders(datasets, cfg["dataloader"])
    train_ds = datasets["train"]

    LOGGER.info("Class to Index Mapping: %s", train_ds.class_to_idx)
    LOGGER.info("Number of images in train dataset: %d", len(datasets["train"]))
    LOGGER.info("Number of images in valid dataset: %d", len(datasets["valid"]))
    LOGGER.info("Number of images in test dataset: %d", len(datasets["test"]))

    images, labels = next(iter(loaders["train"]))
    LOGGER.info("Batch shape: %s, Labels shape: %s", tuple(images.shape), tuple(labels.shape))

    label_examples = labels[: min(8, len(labels))]
    class_names = [train_ds.classes[label.item()] for label in label_examples]
    LOGGER.info("Sample labels: %s", label_examples.tolist())
    LOGGER.info("Sample class names: %s", class_names)

    save_grid = bool(cfg["dataloader"].get("save_sample_batch", False)) or args.save_grid
    if save_grid:
        outputs = prepare_output_dirs("sanity_check", cfg["outputs"]["root_dir"])
        grid = torchvision.utils.make_grid(images[:16], nrow=4)
        grid = grid.cpu()

        # Convert normalized tensor to display range.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        grid = torch.clamp(grid * std + mean, 0.0, 1.0)

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plot_path = Path(outputs["plots_dir"]) / "sample_batch.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved sample grid to: %s", plot_path)


if __name__ == "__main__":
    main()
