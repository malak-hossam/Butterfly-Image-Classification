import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_transforms, get_eval_transforms, get_train_transforms


def _dummy_image() -> Image.Image:
    arr = np.random.randint(0, 255, size=(320, 320, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_train_transforms_output_shape() -> None:
    image = _dummy_image()
    transform = get_train_transforms()
    tensor = transform(image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_eval_transforms_output_shape() -> None:
    image = _dummy_image()
    transform = get_eval_transforms()
    tensor = transform(image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_build_transforms_keys() -> None:
    transforms = build_transforms()
    assert set(transforms.keys()) == {"train", "valid", "test"}
