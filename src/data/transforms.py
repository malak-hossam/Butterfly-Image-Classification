from __future__ import annotations

from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_transforms() -> dict[str, T.Compose]:
    eval_t = get_eval_transforms()
    return {"train": get_train_transforms(), "valid": eval_t, "test": eval_t}
