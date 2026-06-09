from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .constants import CLASS_NAMES


class SmallCNN(nn.Module):
    """Small CNN suitable for a transparent project baseline and CPU deployment."""

    def __init__(self, num_classes: int = len(CLASS_NAMES), dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(
    architecture: str = "small_cnn",
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = False,
) -> nn.Module:
    if architecture == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        original = model.conv1
        model.conv1 = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                model.conv1.weight.copy_(original.weight.mean(dim=1, keepdim=True))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture '{architecture}'. Use 'small_cnn' or 'resnet18'.")
