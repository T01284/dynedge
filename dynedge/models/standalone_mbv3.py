"""Capacity-boosted standalone MBV3-FER variants for post–Step-A optimization.

Variants
  v1       : original Stage-A backbone (stem16 / 24-40-80-160 / head960)
  v3_wide  : ~2× channels + SE + stronger stem (capacity fix per Step-A)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.elastic_mbv3 import ConvBNAct, make_divisible


class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(8, ch // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class InvertedResidualSE(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: float, use_se: bool = True):
        super().__init__()
        self.stride = stride
        hidden = make_divisible(inp * expand_ratio)
        self.use_res = stride == 1 and inp == oup
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(inp, hidden, k=1))
        layers.extend(
            [
                nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.Hardswish(inplace=True),
            ]
        )
        self.expand_dw = nn.Sequential(*layers)
        self.se = SqueezeExcite(hidden) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand_dw(x)
        y = self.se(y)
        y = self.project(y)
        return x + y if self.use_res else y


ARCH_SPECS = {
    # (stem_out, stage_specs[(out, stride, expand), ...], head_mid)
    "v1": {
        "stem": 16,
        "stages": ((24, 2, 3), (40, 2, 3), (80, 2, 4), (160, 2, 6)),
        "head_mid": 960,
        "use_se": False,
        "block": "plain",
    },
    "v3_wide": {
        "stem": 32,
        "stages": ((48, 2, 4), (72, 2, 4), (144, 2, 6), (288, 2, 6)),
        "head_mid": 1280,
        "use_se": True,
        "block": "se",
    },
}


class StandaloneMBV3FER(nn.Module):
    """Fixed-depth FER backbone; `arch` selects capacity recipe."""

    def __init__(
        self,
        num_classes: int,
        depths: tuple[int, int, int, int],
        arch: str = "v1",
        use_se: bool | None = None,
    ):
        super().__init__()
        if arch not in ARCH_SPECS:
            raise ValueError(f"unknown arch={arch}, choose from {list(ARCH_SPECS)}")
        if len(depths) != 4:
            raise ValueError(f"depths must have 4 ints, got {depths}")
        spec = ARCH_SPECS[arch]
        self.arch = arch
        self.depths = depths
        self.num_classes = num_classes

        stem_ch = spec["stem"]
        self.stem = nn.Sequential(
            ConvBNAct(3, stem_ch, k=3, s=2),
            ConvBNAct(stem_ch, stem_ch, k=3, s=1) if arch == "v3_wide" else nn.Identity(),
        )

        # None → arch default; False forces SE-free (Job0-style SE ablation).
        self.use_se = bool(spec["use_se"]) if use_se is None else bool(use_se)
        stages = []
        in_ch = stem_ch
        last_out = stem_ch
        for si, (out_ch, stride, expand) in enumerate(spec["stages"]):
            blocks = []
            d = depths[si]
            if d < 1:
                raise ValueError(f"depth[{si}] must be >=1, got {d}")
            for i in range(d):
                s = stride if i == 0 else 1
                inp = in_ch if i == 0 else out_ch
                if spec["block"] == "se":
                    blocks.append(
                        InvertedResidualSE(inp, out_ch, s, expand, use_se=self.use_se)
                    )
                else:
                    from models.elastic_mbv3 import InvertedResidual

                    blocks.append(InvertedResidual(inp, out_ch, s, expand))
                if i == 0:
                    in_ch = out_ch
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
            last_out = out_ch
        self.stages = nn.ModuleList(stages)
        head_mid = spec["head_mid"]
        self.head = nn.Sequential(
            ConvBNAct(last_out, head_mid, k=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2 if arch == "v3_wide" else 0.0),
            nn.Linear(head_mid, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)

    def key(self) -> str:
        return "d" + "".join(str(d) for d in self.depths)
