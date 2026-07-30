"""BranchyNet-style early-exit ResNet-18 for FER baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ExitHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchyResNet18(nn.Module):
    """ResNet-18 with exits after layer2 / layer3 / final (BranchyNet style)."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.exit2 = ExitHead(128, num_classes)
        self.exit3 = ExitHead(256, num_classes)
        self.exit4 = ExitHead(512, num_classes)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        e2 = self.exit2(x)
        x = self.layer3(x)
        e3 = self.exit3(x)
        x = self.layer4(x)
        e4 = self.exit4(x)
        if return_all:
            return e2, e3, e4
        return e4
