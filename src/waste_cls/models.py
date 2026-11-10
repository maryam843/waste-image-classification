"""The three models compared in this project.

- DNN: the original DNN, unchanged (flatten -> [512, 256, 128] -> 4).
- SmallCNN: 4 conv blocks with BatchNorm, trained from scratch.
- ResNet18Frozen: ImageNet ResNet-18 with a frozen backbone and a new linear head.

This file only depends on torch/torchvision so it can be copied into the Hugging Face Space.
"""

from __future__ import annotations

import torch
from torch import nn

N_CLASSES = 4


class DNN(nn.Module):
    """Fully connected network exactly as in the original version."""

    def __init__(self, input_size: int, num_classes: int = N_CLASSES,
                 hidden_sizes: tuple[int, ...] = (512, 256, 128), dropout: float = 0.5):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(1))


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """4 conv blocks (32-64-128-256 channels), global average pooling, dropout, linear head."""

    def __init__(self, num_classes: int = N_CLASSES, channels: tuple[int, ...] = (32, 64, 128, 256),
                 dropout: float = 0.3):
        super().__init__()
        blocks, prev = [], 3
        for c in channels:
            blocks.append(_conv_block(prev, c))
            prev = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.features[-1][0]  # last conv layer (before its BN/ReLU/pool)


class ResNet18Frozen(nn.Module):
    """ResNet-18 backbone (frozen, kept in eval mode) + trainable linear head."""

    def __init__(self, num_classes: int = N_CLASSES, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(in_features, num_classes)
        self.backbone.eval()

    def train(self, mode: bool = True) -> ResNet18Frozen:
        # Only the head trains; the frozen backbone keeps its BatchNorm statistics.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


MODEL_NAMES = ("dnn", "smallcnn", "resnet18")
DISPLAY_NAMES = {"dnn": "DNN [512,256,128]", "smallcnn": "SmallCNN", "resnet18": "ResNet-18 (frozen)"}


def build_model(name: str, image_size: int = 224, num_classes: int = N_CLASSES, pretrained: bool = True,
                dnn_hidden: tuple[int, ...] = (512, 256, 128), dnn_dropout: float = 0.5) -> nn.Module:
    """Factory: always returns a NEW, freshly initialised model."""
    if name == "dnn":
        return DNN(3 * image_size * image_size, num_classes, tuple(dnn_hidden), dnn_dropout)
    if name == "smallcnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        return ResNet18Frozen(num_classes, pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def export_model(model: nn.Module, arch: str, path, image_size: int, class_names: tuple[str, ...],
                 dnn_hidden: tuple[int, ...] = (512, 256, 128), extra: dict | None = None) -> None:
    """Save weights + everything needed to rebuild the model. Plain tensors/strings only,
    so it can be loaded with torch.load(weights_only=True)."""
    torch.save({
        "arch": arch,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "image_size": int(image_size),
        "class_names": list(class_names),
        "dnn_hidden": list(dnn_hidden),
        "extra": extra or {},
    }, path)


def load_exported(path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a file written by export_model. Never unpickles arbitrary objects (weights_only=True)."""
    blob = torch.load(path, map_location=device, weights_only=True)
    model = build_model(blob["arch"], image_size=blob["image_size"], num_classes=len(blob["class_names"]),
                        pretrained=False, dnn_hidden=tuple(blob.get("dnn_hidden", (512, 256, 128))))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob
