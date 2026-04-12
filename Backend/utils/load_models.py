import torch
from models_architecture import (
    DenseNetCBAMV2,
    ResNet50CBAMV2,
    MobileNetCBAMV2,
    EfficientNetCBAMV2
)

def load_models():

    models = {}

    models["DenseNet121"] = DenseNetCBAMV2()
    models["DenseNet121"].load_state_dict(
        torch.load("models/DenseNet121.pth", map_location="cpu"),
        strict=False
    )

    models["ResNet50"] = ResNet50CBAMV2()
    models["ResNet50"].load_state_dict(
        torch.load("models/ResNet50.pth", map_location="cpu"),
        strict=False
    )

    models["EfficientNetV2S"] = EfficientNetCBAMV2()
    models["EfficientNetV2S"].load_state_dict(
        torch.load("models/EfficientNetV25.pth", map_location="cpu"),
        strict=False
    )

    models["MobileNetV3"] = MobileNetCBAMV2()
    models["MobileNetV3"].load_state_dict(
        torch.load("models/MobileNetV3.pth", map_location="cpu"),
        strict=False
    )

    for model in models.values():
        model.eval()

    return models