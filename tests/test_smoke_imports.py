import torch


def test_smoke_imports() -> None:
    import src.data.datamodule  # noqa: F401
    import src.data.sanity_check  # noqa: F401
    import src.data.transforms  # noqa: F401
    import src.eval.evaluate  # noqa: F401
    import src.infer.predict  # noqa: F401
    import src.models.cnn_model  # noqa: F401
    import src.models.metrics  # noqa: F401
    import src.models.resnet18_model  # noqa: F401
    import src.train.train_cnn  # noqa: F401
    import src.train.train_resnet18  # noqa: F401
    import src.utils.config  # noqa: F401
    import src.utils.logging  # noqa: F401
    import src.utils.paths  # noqa: F401
    import src.utils.seed  # noqa: F401


def test_model_forward_shapes() -> None:
    from src.models.cnn_model import CustomCNN
    from src.models.resnet18_model import build_resnet18

    x = torch.randn(2, 3, 224, 224)

    cnn = CustomCNN(num_classes=10)
    y_cnn = cnn(x)
    assert y_cnn.shape == (2, 10)

    resnet = build_resnet18(num_classes=10, freeze_backbone=True, pretrained=False)
    y_res = resnet(x)
    assert y_res.shape == (2, 10)
