"""Elastic MobileNetV3-style FER supernet with early-exit heads.

v1 (arch=v1, slimmable=False): depth-only; width tag ignored (formal v1 ckpt).
v2/slimmable (slimmable=True, arch=v1): width channel-slicing + Slimmable BN.
v3_wide (arch=v3_wide, slimmable=False): wider+SE backbone, depth-only elasticity.
v4 (arch=v3_wide, slimmable=True): channel-slicing + SE-aware Slimmable IR + res/gray.

Smoke test::

    python -c "from models.elastic_mbv3 import ElasticFERSuperNet; \
m=ElasticFERSuperNet(8,slimmable=True,arch='v3_wide'); \
c=m.random_config(); import torch; \
y=m(torch.randn(2,3,c.img_size,c.img_size), subnet=c); print(c, y.shape)"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slimmable_bn import (
    SlimmableBatchNorm2d,
    bn_channels_expand,
    bn_channels_scaled,
    set_model_width,
)


def make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


WIDTH_CHOICES = (0.35, 0.5, 0.75, 1.0)
DEPTH_CHOICES = (2, 3, 4)
RES_CHOICES = (64, 96, 112)

# Depth-elastic backbone recipes (shared with standalone_mbv3 ARCH_SPECS).
ARCH_SPECS = {
    "v1": {
        "stem": 16,
        "stem_extra": False,
        "stages": ((24, 2, 3), (40, 2, 3), (80, 2, 4), (160, 2, 6)),
        "head_mid": 960,
        "use_se": False,
        "dropout": 0.0,
    },
    "v3_wide": {
        "stem": 32,
        "stem_extra": True,
        "stages": ((48, 2, 4), (72, 2, 4), (144, 2, 6), (288, 2, 6)),
        "head_mid": 1280,
        "use_se": True,
        "dropout": 0.2,
    },
}


def _slice_conv2d(conv: nn.Conv2d, x: torch.Tensor, in_ch: int, out_ch: int) -> torch.Tensor:
    if conv.groups == 1:
        w = conv.weight[:out_ch, :in_ch]
        b = None if conv.bias is None else conv.bias[:out_ch]
        return F.conv2d(x, w, b, conv.stride, conv.padding, conv.dilation, 1)
    w = conv.weight[:out_ch]
    b = None if conv.bias is None else conv.bias[:out_ch]
    return F.conv2d(x, w, b, conv.stride, conv.padding, conv.dilation, out_ch)


class ConvBNAct(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, act=True):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(inp, oup, k, s, pad, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = nn.Hardswish(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SlimmableConvBNAct(nn.Module):
    def __init__(
        self,
        max_inp: int,
        max_oup: int,
        width_list: Sequence[float],
        k: int = 3,
        s: int = 1,
        act: bool = True,
        *,
        oup_channels: dict[float, int] | None = None,
        fixed_in_ch: int | None = None,
    ):
        super().__init__()
        pad = k // 2
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.fixed_in_ch = fixed_in_ch  # e.g. RGB stem always 3
        self.conv = nn.Conv2d(max_inp, max_oup, k, s, pad, bias=False)
        ch = oup_channels or bn_channels_scaled(max_oup, width_list, make_divisible)
        self.bn = SlimmableBatchNorm2d(ch)
        self.act = nn.Hardswish(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_ch = self.fixed_in_ch if self.fixed_in_ch is not None else x.size(1)
        out_ch = self.bn.channels_for()
        y = _slice_conv2d(self.conv, x, in_ch, out_ch)
        return self.act(self.bn(y))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
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
                nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


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
    """IR block with optional SE (used by arch=v3_wide)."""

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


class SlimmableInvertedResidual(nn.Module):
    def __init__(
        self,
        max_inp: int,
        max_oup: int,
        stride: int,
        expand_ratio: int,
        width_list: Sequence[float],
    ):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.max_hidden = make_divisible(max_inp * expand_ratio)
        self.width_list = tuple(width_list)
        self.inp_ch = bn_channels_scaled(max_inp, width_list, make_divisible)
        self.hid_ch = bn_channels_expand(max_inp, expand_ratio, width_list, make_divisible)
        self.oup_ch = bn_channels_scaled(max_oup, width_list, make_divisible)

        self.expand: SlimmableConvBNAct | None
        if expand_ratio != 1:
            self.expand = SlimmableConvBNAct(
                max_inp,
                self.max_hidden,
                width_list,
                k=1,
                s=1,
                act=True,
                oup_channels=self.hid_ch,
            )
        else:
            self.expand = None

        self.dw = nn.Conv2d(
            self.max_hidden,
            self.max_hidden,
            3,
            stride,
            1,
            groups=self.max_hidden,
            bias=False,
        )
        self.dw_bn = SlimmableBatchNorm2d(self.hid_ch)
        self.act = nn.Hardswish(inplace=True)
        self.project = nn.Conv2d(self.max_hidden, max_oup, 1, 1, 0, bias=False)
        self.project_bn = SlimmableBatchNorm2d(self.oup_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # active width already set on all Slimmable BNs
        w = self.project_bn.active_width
        in_ch = self.inp_ch[w]
        out_ch = self.oup_ch[w]
        hid_ch = self.hid_ch[w]
        use_res = self.stride == 1 and in_ch == out_ch

        if self.expand is not None:
            y = self.expand(x)
        else:
            y = x
        y = _slice_conv2d(self.dw, y, hid_ch, hid_ch)
        y = self.act(self.dw_bn(y))
        y = _slice_conv2d(self.project, y, hid_ch, out_ch)
        y = self.project_bn(y)
        return x + y if use_res else y


class SlimmableSqueezeExcite(nn.Module):
    """Channel-sliced SE: shared Linear weights, active width from input channels."""

    def __init__(self, max_ch: int, width_list: Sequence[float], reduction: int = 4):
        super().__init__()
        self.max_ch = max_ch
        self.reduction = reduction
        self.width_list = tuple(width_list)
        self.max_mid = max(8, max_ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(max_ch, self.max_mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.max_mid, max_ch)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        mid = max(8, c // self.reduction)
        if mid > self.max_mid or c > self.max_ch:
            raise RuntimeError(f"SE slice out of range: c={c} mid={mid} max={self.max_ch}/{self.max_mid}")
        y = self.pool(x).flatten(1)
        y = F.linear(y, self.fc1.weight[:mid, :c], self.fc1.bias[:mid])
        y = self.act(y)
        y = F.linear(y, self.fc2.weight[:c, :mid], self.fc2.bias[:c])
        # unsqueeze (not view) so ONNX keeps dynamic channel dim under channel-slicing.
        y = self.gate(y).unsqueeze(-1).unsqueeze(-1)
        return x * y


class SlimmableInvertedResidualSE(nn.Module):
    """Slimmable IR with SE between dw/act and project (v3_wide / v4)."""

    def __init__(
        self,
        max_inp: int,
        max_oup: int,
        stride: int,
        expand_ratio: int,
        width_list: Sequence[float],
        use_se: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.max_inp = max_inp
        self.max_oup = max_oup
        self.max_hidden = make_divisible(max_inp * expand_ratio)
        self.width_list = tuple(width_list)
        self.inp_ch = bn_channels_scaled(max_inp, width_list, make_divisible)
        self.hid_ch = bn_channels_expand(max_inp, expand_ratio, width_list, make_divisible)
        self.oup_ch = bn_channels_scaled(max_oup, width_list, make_divisible)

        self.expand: SlimmableConvBNAct | None
        if expand_ratio != 1:
            self.expand = SlimmableConvBNAct(
                max_inp,
                self.max_hidden,
                width_list,
                k=1,
                s=1,
                act=True,
                oup_channels=self.hid_ch,
            )
        else:
            self.expand = None

        self.dw = nn.Conv2d(
            self.max_hidden,
            self.max_hidden,
            3,
            stride,
            1,
            groups=self.max_hidden,
            bias=False,
        )
        self.dw_bn = SlimmableBatchNorm2d(self.hid_ch)
        self.act = nn.Hardswish(inplace=True)
        self.se: nn.Module = (
            SlimmableSqueezeExcite(self.max_hidden, width_list) if use_se else nn.Identity()
        )
        self.project = nn.Conv2d(self.max_hidden, max_oup, 1, 1, 0, bias=False)
        self.project_bn = SlimmableBatchNorm2d(self.oup_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.project_bn.active_width
        in_ch = self.inp_ch[w]
        out_ch = self.oup_ch[w]
        hid_ch = self.hid_ch[w]
        use_res = self.stride == 1 and in_ch == out_ch

        if self.expand is not None:
            y = self.expand(x)
        else:
            y = x
        y = _slice_conv2d(self.dw, y, hid_ch, hid_ch)
        y = self.act(self.dw_bn(y))
        y = self.se(y)
        y = _slice_conv2d(self.project, y, hid_ch, out_ch)
        y = self.project_bn(y)
        return x + y if use_res else y


class ExitHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        mid = max(64, in_ch // 2)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, mid),
            nn.Hardswish(inplace=True),
            nn.Linear(mid, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SlimmableExitHead(nn.Module):
    def __init__(self, max_in: int, num_classes: int, width_list: Sequence[float]):
        super().__init__()
        self.max_in = max_in
        self.width_list = tuple(width_list)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.max_mid = max(64, max_in // 2)
        self.fc1 = nn.Linear(max_in, self.max_mid)
        self.act = nn.Hardswish(inplace=True)
        self.fc2 = nn.Linear(self.max_mid, num_classes)
        self.in_ch = bn_channels_scaled(max_in, width_list, make_divisible)
        self.mid_ch = {w: max(64, make_divisible(self.max_mid * w)) for w in width_list}

    def forward(self, x: torch.Tensor, width: float) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        in_ch = self.in_ch[float(width)]
        mid = self.mid_ch[float(width)]
        if x.size(1) != in_ch:
            raise RuntimeError(f"exit expects {in_ch}, got {x.size(1)}")
        h = F.linear(x, self.fc1.weight[:mid, :in_ch], self.fc1.bias[:mid])
        h = self.act(h)
        return F.linear(h, self.fc2.weight[:, :mid], self.fc2.bias)


@dataclass(frozen=True)
class SubnetConfig:
    width: float
    depths: tuple[int, int, int, int]
    img_size: int = 112
    gray: bool = False


class ElasticFERSuperNet(nn.Module):
    # Legacy class attr = v1 (kept for external imports / older scripts).
    STAGE_SPECS = ARCH_SPECS["v1"]["stages"]

    def __init__(
        self,
        num_classes: int = 8,
        max_depth: int = 4,
        slimmable: bool = False,
        width_list: Sequence[float] = WIDTH_CHOICES,
        arch: str = "v1",
    ):
        super().__init__()
        if arch not in ARCH_SPECS:
            raise ValueError(f"unknown arch={arch}, choose from {list(ARCH_SPECS)}")
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.slimmable = slimmable
        self.width_list = tuple(width_list)
        self.arch = arch
        spec = ARCH_SPECS[arch]
        self.STAGE_SPECS = spec["stages"]
        self.stem_ch = int(spec["stem"])
        self.head_mid = int(spec["head_mid"])
        self.use_se = bool(spec["use_se"])
        self.dropout_p = float(spec["dropout"])
        # Which stage indices (0-based, after that stage) emit early-exit heads.
        # Default (1,2,3) = three mid exits + final → K=4 total exits.
        # Set via set_enabled_exit_stages() for K-ablation without rebuilding weights.
        self.enabled_exit_stages: tuple[int, ...] = (1, 2, 3)

        if not slimmable:
            stem_layers: list[nn.Module] = [ConvBNAct(3, self.stem_ch, k=3, s=2)]
            if spec["stem_extra"]:
                stem_layers.append(ConvBNAct(self.stem_ch, self.stem_ch, k=3, s=1))
            self.stem = nn.Sequential(*stem_layers)
            self.stages = nn.ModuleList()
            in_ch = self.stem_ch
            for out_ch, stride, expand in self.STAGE_SPECS:
                blocks = nn.ModuleList()
                for i in range(max_depth):
                    s = stride if i == 0 else 1
                    inp = in_ch if i == 0 else out_ch
                    if self.use_se:
                        blocks.append(InvertedResidualSE(inp, out_ch, s, expand, use_se=True))
                    else:
                        blocks.append(InvertedResidual(inp, out_ch, s, expand))
                    if i == 0:
                        in_ch = out_ch
                self.stages.append(blocks)
                in_ch = out_ch
            self.stage_out_channels = tuple(s[0] for s in self.STAGE_SPECS)
            # Early exits after stages 1/2/3 (0-indexed).
            self.exit2 = ExitHead(self.stage_out_channels[1], num_classes)
            self.exit3 = ExitHead(self.stage_out_channels[2], num_classes)
            self.exit4 = ExitHead(self.stage_out_channels[3], num_classes)
            head_layers: list[nn.Module] = [
                ConvBNAct(self.stage_out_channels[3], self.head_mid, k=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Hardswish(inplace=True),
            ]
            if self.dropout_p > 0:
                head_layers.append(nn.Dropout(self.dropout_p))
            head_layers.append(nn.Linear(self.head_mid, num_classes))
            self.final = nn.Sequential(*head_layers)
            self.final_pw = None
            self.final_fc = None
            return

        # Slimmable path: v1 or v3_wide (channel-slicing + per-width BN).
        stem_oup = bn_channels_scaled(self.stem_ch, self.width_list, make_divisible)
        stem_layers_s: list[nn.Module] = [
            SlimmableConvBNAct(
                3,
                self.stem_ch,
                self.width_list,
                k=3,
                s=2,
                act=True,
                oup_channels=stem_oup,
                fixed_in_ch=3,
            )
        ]
        if spec["stem_extra"]:
            stem_layers_s.append(
                SlimmableConvBNAct(
                    self.stem_ch,
                    self.stem_ch,
                    self.width_list,
                    k=3,
                    s=1,
                    act=True,
                    oup_channels=stem_oup,
                )
            )
        self.stem = nn.Sequential(*stem_layers_s)
        self.stages = nn.ModuleList()
        in_ch = self.stem_ch
        for out_ch, stride, expand in self.STAGE_SPECS:
            blocks = nn.ModuleList()
            for i in range(max_depth):
                s = stride if i == 0 else 1
                inp = in_ch if i == 0 else out_ch
                if self.use_se:
                    blocks.append(
                        SlimmableInvertedResidualSE(
                            inp, out_ch, s, expand, self.width_list, use_se=True
                        )
                    )
                else:
                    blocks.append(
                        SlimmableInvertedResidual(inp, out_ch, s, expand, self.width_list)
                    )
                if i == 0:
                    in_ch = out_ch
            self.stages.append(blocks)
            in_ch = out_ch
        self.stage_out_channels = tuple(s[0] for s in self.STAGE_SPECS)
        self.exit2 = SlimmableExitHead(self.stage_out_channels[1], num_classes, self.width_list)
        self.exit3 = SlimmableExitHead(self.stage_out_channels[2], num_classes, self.width_list)
        self.exit4 = SlimmableExitHead(self.stage_out_channels[3], num_classes, self.width_list)
        last = self.stage_out_channels[3]
        self.final_pw = SlimmableConvBNAct(
            last, self.head_mid, self.width_list, k=1, s=1, act=True
        )
        self.final_fc = nn.Linear(self.head_mid, num_classes)
        self.final = None

    def set_enabled_exit_stages(self, stages: Sequence[int]) -> None:
        """Enable mid-exit heads after given stage indices (1/2/3). Final always on."""
        allowed = {1, 2, 3}
        cleaned = tuple(sorted({int(s) for s in stages if int(s) in allowed}))
        self.enabled_exit_stages = cleaned

    def set_num_exits(self, k: int) -> None:
        """Set exit-head budget by total exit count including final.

        Code mapping (training tags): K∈{1,2,3,4} → mid stages (), (2,), (1,2), (1,2,3).
        Paper ablation uses K∈{1,2,3} counting intermediate exits at full capacity:
        paper K=1 ↔ code 1; paper K=2 ↔ code 2; paper K=3 (=v4) ↔ code 4 / tag k4.
        """
        mapping = {
            1: (),
            2: (2,),
            3: (1, 2),
            4: (1, 2, 3),
        }
        if k not in mapping:
            raise ValueError(f"num_exits={k} not in {sorted(mapping)}")
        self.set_enabled_exit_stages(mapping[k])

    def filter_exit_outputs(
        self, outs: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Map full (e2,e3,e4,final) → enabled mid exits + final."""
        if len(outs) != 4:
            return tuple(outs)
        stage_to_idx = {1: 0, 2: 1, 3: 2}
        selected = [outs[stage_to_idx[s]] for s in self.enabled_exit_stages]
        selected.append(outs[-1])
        return tuple(selected)

    @staticmethod
    def max_config() -> SubnetConfig:
        return SubnetConfig(width=1.0, depths=(4, 4, 4, 4), img_size=112, gray=False)

    @staticmethod
    def min_config() -> SubnetConfig:
        # width ignored when not slimmable (depth-only); img_size/gray used by v4 trainer.
        return SubnetConfig(width=0.35, depths=(2, 2, 2, 2), img_size=64, gray=True)

    @staticmethod
    def random_config(rng: torch.Generator | None = None) -> SubnetConfig:
        def pick(opts):
            if rng is None:
                i = int(torch.randint(0, len(opts), (1,)).item())
            else:
                i = int(torch.randint(0, len(opts), (1,), generator=rng).item())
            return opts[i]

        depths = tuple(pick(DEPTH_CHOICES) for _ in range(4))
        img_size = int(pick(RES_CHOICES))
        return SubnetConfig(
            width=float(pick(WIDTH_CHOICES)),
            depths=depths,  # type: ignore[arg-type]
            img_size=img_size,
            gray=(img_size == 64),
        )

    def scaled_channels(self, max_ch: int, width: float) -> int:
        return make_divisible(max_ch * width)

    def estimate_flops(self, subnet: SubnetConfig, img_size: int | None = None) -> float:
        if img_size is None:
            img_size = subnet.img_size
        w = subnet.width if self.slimmable else 1.0
        stem_c = self.scaled_channels(self.stem_ch, w)
        h = img_size // 2
        macs = 3 * stem_c * 9 * h * h
        if ARCH_SPECS[self.arch]["stem_extra"]:
            macs += stem_c * stem_c * 9 * h * h
        in_ch = stem_c
        for si, (out_ch_max, stride, expand) in enumerate(self.STAGE_SPECS):
            out_ch = self.scaled_channels(out_ch_max, w)
            for bi in range(subnet.depths[si]):
                s = stride if bi == 0 else 1
                hid = make_divisible(in_ch * expand)
                oh = h // s
                macs += in_ch * hid * oh * oh
                macs += hid * 9 * oh * oh
                macs += hid * out_ch * oh * oh
                h = oh
                in_ch = out_ch
        mid = self.scaled_channels(self.head_mid, w)
        macs += in_ch * mid * h * h
        macs += mid * self.num_classes
        return float(macs)

    def forward(
        self,
        x: torch.Tensor,
        subnet: SubnetConfig | None = None,
        return_all_exits: bool = False,
    ):
        if subnet is None:
            subnet = self.max_config()

        if not self.slimmable:
            x = self.stem(x)
            exits = []
            for si, blocks in enumerate(self.stages):
                for bi in range(subnet.depths[si]):
                    x = blocks[bi](x)
                if si == 1:
                    exits.append(self.exit2(x))
                elif si == 2:
                    exits.append(self.exit3(x))
                elif si == 3:
                    exits.append(self.exit4(x))
            logits_final = self.final(x)
            if return_all_exits:
                return self.filter_exit_outputs(
                    (exits[0], exits[1], exits[2], logits_final)
                )
            return logits_final

        set_model_width(self, subnet.width)
        x = self.stem(x)
        exits = []
        for si, blocks in enumerate(self.stages):
            for bi in range(subnet.depths[si]):
                x = blocks[bi](x)
            if si == 1:
                exits.append(self.exit2(x, subnet.width))
            elif si == 2:
                exits.append(self.exit3(x, subnet.width))
            elif si == 3:
                exits.append(self.exit4(x, subnet.width))

        x = self.final_pw(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.dropout_p > 0 and self.training:
            x = F.dropout(x, p=self.dropout_p)
        mid = self.scaled_channels(self.head_mid, subnet.width)
        logits_final = F.linear(x, self.final_fc.weight[:, :mid], self.final_fc.bias)
        if return_all_exits:
            return self.filter_exit_outputs(
                (exits[0], exits[1], exits[2], logits_final)
            )
        return logits_final


def exit_loss(
    logits_list: Sequence[torch.Tensor],
    y: torch.Tensor,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    n = len(logits_list)
    if weights is None:
        # ramp toward final; length adapts to K
        if n == 1:
            weights = (1.0,)
        elif n == 2:
            weights = (0.7, 1.0)
        elif n == 3:
            weights = (0.5, 0.8, 1.0)
        else:
            weights = (0.5, 0.7, 0.9, 1.0)
    if len(weights) != n:
        # truncate / pad last=1.0
        w = list(weights[:n])
        while len(w) < n:
            w.append(1.0)
        w[-1] = 1.0
        weights = tuple(w)
    loss = 0.0
    for logit, w in zip(logits_list, weights):
        loss = loss + w * F.cross_entropy(logit, y)
    return loss  # type: ignore[return-value]


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 4.0) -> torch.Tensor:
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits.detach() / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)
