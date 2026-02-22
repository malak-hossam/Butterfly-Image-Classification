from __future__ import annotations

import logging

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


LOGGER = logging.getLogger(__name__)


def build_resnet18(
    num_classes: int,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Falling back to randomly initialized ResNet18: %s", exc)
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
