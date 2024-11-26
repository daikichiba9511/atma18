from typing import Any, TypeAlias, cast

import timm
import torch
import torch.nn as nn
from timm.utils import ModelEmaV3


class FFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Atma18VisionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_features: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=9)
        self.n_features = self.encoder.num_features
        self.n_label = 18
        self.fc = nn.Linear(self.n_features, self.n_label)
        self.neck = FFN(self.n_features + in_features, self.n_features // 2, self.n_features)
        self.fc_aux1 = nn.Linear(self.n_features, 4)
        self.fc_aux_cls_blinker = nn.Linear(self.n_features, 2)
        self.fc_aux_cls_brake = nn.Linear(self.n_features, 1)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encoder(x)
        x = self.neck(torch.concat([x, features], dim=1))
        logits = self.fc(x)
        logits_aux1 = self.fc_aux1(x)
        logits_aux_cls_blinker = self.fc_aux_cls_blinker(x)
        logits_aux_cls_brake = self.fc_aux_cls_brake(x)
        return {
            "logits": logits,
            "logits_aux1": logits_aux1,
            "logits_aux_cls_blinker": logits_aux_cls_blinker,
            "logits_aux_cls_brake": logits_aux_cls_brake,
        }


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

    # model_params: dict[str, Any] = {"model_name": "resnet18", "pretrained": False}
    # model, ema_model = get_model(model_name, model_params=model_params)
    # summary(model, input_size=(8, 9, 64, 128))

    model_params: dict[str, Any] = {
        "model_name": "convnext_tiny.fb_in22k_ft_in1k",
        "pretrained": False,
        "in_features": 9,
    }
    model, ema_model = get_model(model_name, model_params=model_params)
    model, ema_model = model.cpu(), ema_model.cpu()
    # summary(model, input_size=(8, 9, 64, 128))

    # model_params: dict[str, Any] = {"model_name": "eva02_base_patch14_224.mim_in22k", "pretrained": False}
    # model, ema_model = get_model(model_name, model_params=model_params)
    # model, ema_model = model.cpu(), ema_model.cpu()
    # summary(model, input_size=(8, 9, 224, 224), device="cpu")

    x = torch.randn(8, 9, 224, 224)
    features = torch.randn(8, 9)
    out = model(x, features)
    for k, v in out.items():
        print(f"{k = }: {v.shape = }")

    print("Done!")
