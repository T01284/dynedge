"""DDAMFN-style FER backbone: dual CoordAtt heads on ResNet-34 features.

Faithful to SainingZhang/DDAMFN attention pattern without requiring
MixedFeatureNet MS-Celeb weights (often unavailable on AutoDL).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class HSwish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3, inplace=True) / 6


class CoordAtt(nn.Module):
    def __init__(self, inp: int, oup: int, groups: int = 32):
        super().__init__()
        self.linear_h = LinearBlock(inp, inp, groups=inp, kernel=(1, 7), padding=(0, 0))
        self.linear_w = LinearBlock(inp, inp, groups=inp, kernel=(7, 1), padding=(0, 0))
        mip = max(8, inp // groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, bias=False)
        self.act = HSwish()

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.linear_h(x)
        x_w = self.linear_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h).sigmoid().expand(-1, -1, h, w)
        x_w = self.conv3(x_w).sigmoid().expand(-1, -1, h, w)
        return x_w * x_h


class CoordAttHead(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.att = CoordAtt(channels, channels)

    def forward(self, x):
        return self.att(x)


class DDAMFN(nn.Module):
    """Dual-Direction Attention Mixed Feature Network (style)."""

    def __init__(self, num_classes: int = 7, num_head: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        # Keep up to layer4; drop avgpool+fc. Output: B×512×7×7 at 224, B×512×4×4 at 112.
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # CoordAtt uses 7×1 / 1×7 kernels — force spatial size to 7×7 for any img_size.
        self.force7 = nn.AdaptiveAvgPool2d(7)
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, f"cat_head{i}", CoordAttHead(512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.force7(self.features(x))
        heads = [getattr(self, f"cat_head{i}")(feat) for i in range(self.num_head)]
        att = heads[0]
        for h in heads[1:]:
            att = torch.max(att, h)
        y = feat * att
        y = self.pool(y).flatten(1)
        return self.fc(y)
