#!/usr/bin/env python3
"""Export a distilled elastic subnet to a static ONNX graph (for ESP-PPQ / ESP-DL).

Wraps ElasticFERSuperNet at a fixed SubnetConfig so the ONNX graph has no
control-flow. Optionally replaces Hardswish→ReLU for friendlier PTQ.

Example:
  python tools/export_subnet_onnx.py \\
    --ckpt results/from_autodl/subnets/subnet_ferplus_d2222_seed42_v1_best.pt \\
    --depths 2,2,2,2 --img-size 64 --relu-for-deploy \\
    --out deploy/mcu_espdl/models/d2222_ferplus_64.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig


def parse_depths(s: str) -> tuple[int, int, int, int]:
    parts = [int(x) for x in s.replace("x", ",").split(",")]
    if len(parts) != 4:
        raise ValueError(f"depths must be 4 ints, got {s}")
    return parts[0], parts[1], parts[2], parts[3]


def hardswish_to_relu(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Hardswish):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            hardswish_to_relu(child)


class StaticSubnet(nn.Module):
    def __init__(self, net: ElasticFERSuperNet, cfg: SubnetConfig):
        super().__init__()
        self.net = net
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x, subnet=self.cfg, return_all_exits=False)


def remap_v1_stem_keys(state: dict) -> dict:
    """Map legacy `stem.*` (ConvBNAct) → current `stem.0.*` (Sequential)."""
    if any(k.startswith("stem.0.") for k in state):
        return state
    if not any(k.startswith("stem.") for k in state):
        return state
    out = {}
    for k, v in state.items():
        if k.startswith("stem."):
            out["stem.0." + k[len("stem.") :]] = v
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--depths", type=str, default="2,2,2,2")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--arch", type=str, default="", choices=["", "v1", "v3_wide"])
    ap.add_argument(
        "--slimmable",
        action="store_true",
        help="channel-slicing + per-width BN (required for v4 width!=1.0)",
    )
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--gray", action="store_true", help="mark gray→3ch subnet (metadata)")
    ap.add_argument("--num-classes", type=int, default=0, help="0 = infer from ckpt")
    ap.add_argument("--relu-for-deploy", action="store_true", help="Hardswish→ReLU")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--output-name", type=str, default="logits", help="ONNX output blob name")
    args = ap.parse_args()

    depths = parse_depths(args.depths)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = args.arch or ckpt.get("arch") or (ckpt.get("args") or {}).get("arch") or "v1"
    # Infer slimmable from ckpt when not explicitly set via flag presence is awkward;
    # prefer explicit --slimmable, else ckpt fields.
    slimmable = bool(
        args.slimmable
        or ckpt.get("slimmable")
        or (ckpt.get("args") or {}).get("slimmable")
    )
    if args.width != 1.0 and not slimmable:
        raise SystemExit("width!=1.0 requires --slimmable (or a slimmable ckpt)")
    cfg = SubnetConfig(
        width=args.width,
        depths=depths,
        img_size=args.img_size,
        gray=args.gray,
    )
    classes = ckpt.get("classes")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = remap_v1_stem_keys(state)
    num_classes = args.num_classes
    if not num_classes:
        if classes:
            num_classes = len(classes)
        else:
            # infer from classifier weight (final head)
            for key in ("final.5.weight", "final_fc.weight", "classifier.3.weight"):
                if key in state:
                    num_classes = int(state[key].shape[0])
                    break
            else:
                num_classes = 8

    net = ElasticFERSuperNet(num_classes=num_classes, arch=arch, slimmable=slimmable)
    net.load_state_dict(state, strict=True)
    net.eval()
    if args.relu_for_deploy:
        hardswish_to_relu(net)
        print("replaced Hardswish → ReLU")

    wrapped = StaticSubnet(net, cfg).eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        logits = wrapped(dummy)
    print(f"arch={arch} slimmable={slimmable} cfg={cfg} logits={tuple(logits.shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=[args.output_name],
        opset_version=args.opset,
        dynamo=False,
    )
    meta = {
        "ckpt": str(args.ckpt),
        "arch": arch,
        "slimmable": slimmable,
        "depths": list(depths),
        "width": args.width,
        "img_size": args.img_size,
        "gray": args.gray,
        "num_classes": num_classes,
        "relu_for_deploy": args.relu_for_deploy,
        "onnx": str(args.out),
        "classes": classes,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
