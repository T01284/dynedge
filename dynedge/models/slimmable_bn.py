"""Slimmable BatchNorm: one BN stats bank per width choice.

Shared conv weights + per-width BN (US-Net / slimmable-networks).
Active width selected via ``set_active_width`` before forward.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn


def width_key(width: float) -> str:
    return f"w{int(round(width * 100)):03d}"


class SlimmableBatchNorm2d(nn.Module):
    """Per-width BatchNorm2d with explicit channel counts (not always max*w)."""

    def __init__(
        self,
        channels_per_width: Mapping[float, int],
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.width_list = tuple(sorted(float(w) for w in channels_per_width.keys()))
        self.channels_per_width = {float(w): int(c) for w, c in channels_per_width.items()}
        self.active_width = 1.0 if 1.0 in self.channels_per_width else self.width_list[-1]
        self.bns = nn.ModuleDict(
            {
                width_key(w): nn.BatchNorm2d(self.channels_per_width[w], eps=eps, momentum=momentum)
                for w in self.width_list
            }
        )

    def set_active_width(self, width: float) -> None:
        w = float(width)
        if w not in self.channels_per_width:
            raise ValueError(f"width={width} not in {self.width_list}")
        self.active_width = w

    def channels_for(self, width: float | None = None) -> int:
        w = self.active_width if width is None else float(width)
        return self.channels_per_width[w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn = self.bns[width_key(self.active_width)]
        if x.size(1) != bn.num_features:
            raise RuntimeError(
                f"BN expects {bn.num_features} ch, got {x.size(1)} (width={self.active_width})"
            )
        return bn(x)


def bn_channels_scaled(max_ch: int, width_list: Sequence[float], make_divisible_fn) -> dict[float, int]:
    """Standard max*width channel schedule (stem / project / stage outs)."""
    return {float(w): make_divisible_fn(max_ch * w) for w in width_list}


def bn_channels_expand(
    max_inp: int,
    expand: int,
    width_list: Sequence[float],
    make_divisible_fn,
) -> dict[float, int]:
    """Hidden channels = make_divisible(make_divisible(max_inp*w) * expand)."""
    out: dict[float, int] = {}
    for w in width_list:
        inp = make_divisible_fn(max_inp * w)
        out[float(w)] = make_divisible_fn(inp * expand)
    return out


def iter_slimmable_bns(module: nn.Module) -> Iterable[SlimmableBatchNorm2d]:
    for m in module.modules():
        if isinstance(m, SlimmableBatchNorm2d):
            yield m


def set_model_width(module: nn.Module, width: float) -> None:
    for m in iter_slimmable_bns(module):
        m.set_active_width(width)
