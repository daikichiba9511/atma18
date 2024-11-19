from typing import Any, TypeAlias, cast

import timm
import torch
import torch.nn as nn
from timm.utils import ModelEmaV3


class Atma18VisionModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=9)
        self.n_features = self.encoder.num_features
        self.n_label = 18
        self.fc = nn.Linear(self.n_features, self.n_label)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.fc(x)
        return x


Models: TypeAlias = Atma18VisionModel


def get_model(model_name: str, model_params: dict[str, Any], decay: float = 0.995) -> tuple[Models, ModelEmaV3]:
    if model_name == "Atma18VisionModel":
        model = Atma18VisionModel(**model_params)
        ema_model = ModelEmaV3(model, decay=decay)
        return model, ema_model
    raise ValueError(f"Unknown model name: {model_name}")


def compile_models(
    model: Models, ema_model: ModelEmaV3, compile_mode: str = "max-autotune", dynamic: bool = False
) -> tuple[Models, ModelEmaV3]:
    compiled_model = torch.compile(model, mode=compile_mode, dynamic=dynamic)
    compiled_model = cast(Models, compiled_model)
    compiled_ema_model = torch.compile(ema_model, mode=compile_mode, dynamic=dynamic)
    compiled_ema_model = cast(ModelEmaV3, compiled_ema_model)
    return compiled_model, compiled_ema_model


if __name__ == "__main__":
    from torchinfo import summary

    model_name = "Atma18VisionModel"
    model_params: dict[str, Any] = {"model_name": "resnet18", "pretrained": True}
    x = torch.randn(8, 9, 64, 128)

    model, ema_model = get_model(model_name, model_params=model_params)
    y = model(x)
    print(y.shape)
    summary(model, input_size=(8, 9, 64, 128))
    print("Done!")
