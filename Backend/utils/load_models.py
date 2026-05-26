import torch
from models_architecture import DenseNetCBAMV2


def load_models():
    model = DenseNetCBAMV2()
    model.load_state_dict(
        torch.load("models/DenseNet121.pth", map_location="cpu"),
        strict=False,
    )
    model.eval()
    return model