#!/usr/bin/env python3
"""B10/O1: stratified subnet sample for the Orin TensorRT sweep, exported from the supernet.

12 strata (width x resolution), 15 random depth tuples each (seed 2026), plus the 13
architectures already benched in B0. Latency does not depend on distilled weights, so every
subnet is sliced out of the FER+ supernet and exported to a static ONNX graph.

Outputs: results/stageB/orin_sweep/sample.json, results/stageB/orin_sweep/onnx/<key>.onnx
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402
from tools.export_subnet_onnx import StaticSubnet  # noqa: E402

WIDTHS = (0.35, 0.5, 0.75, 1.0)
RES = (64, 96, 112)
DEPTH_TUPLES = list(itertools.product((2, 3, 4), repeat=4))
OUT = ROOT / "results/stageB/orin_sweep"


def key_of(w: float, d, img: int) -> str:
    return f"w{w:g}_d{''.join(map(str, d))}_s{img}{'g' if img == 64 else 'c'}"


def parse(key: str):
    w, d, s = key.split("_")
    return float(w[1:]), tuple(int(c) for c in d[1:]), int(s[1:-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "results/from_autodl/supernet/supernet_ferplus_seed42_v4_best.pt")
    ap.add_argument("--per-stratum", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--all", action="store_true", help="also export every subnet of the space (for O4 fronts)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    keys = []
    for w in WIDTHS:
        for img in RES:
            for d in rng.sample(DEPTH_TUPLES, args.per_stratum):
                keys.append(key_of(w, d, img))
    with (ROOT / "results/stageB/orin_trt_extended.csv").open(encoding="utf-8") as f:
        b0 = sorted({r["arch"].replace("w1.0_", "w1_") for r in csv.DictReader(f)})
    keys = sorted(set(keys) | set(b0))
    order = keys[:]
    rng.shuffle(order)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "onnx").mkdir(exist_ok=True)
    import pathlib

    posix = pathlib.PosixPath
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath  # the checkpoint pickles Linux paths
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = posix
    net = ElasticFERSuperNet(num_classes=8, slimmable=True, arch="v3_wide")
    net.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    net.eval()
    todo = order + sorted({key_of(w, d, r) for w in WIDTHS for d in DEPTH_TUPLES for r in RES} - set(order)) \
        if args.all else order
    for i, key in enumerate(todo):
        out = OUT / "onnx" / f"{key}.onnx"
        if out.is_file():
            continue
        w, d, img = parse(key)
        cfg = SubnetConfig(width=w, depths=d, img_size=img, gray=img == 64)
        with torch.no_grad():
            torch.onnx.export(StaticSubnet(net, cfg).eval(), torch.randn(1, 3, img, img), str(out),
                              input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
        print(f"{i + 1}/{len(todo)} {key} {out.stat().st_size / 1e6:.1f} MB", flush=True)
    (OUT / "sample.json").write_text(json.dumps(
        {"seed": args.seed, "per_stratum": args.per_stratum, "n": len(order), "b0_keys": b0,
         "order": order}, indent=2), encoding="utf-8")
    print(f"sample n={len(order)} (b0 {len(b0)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
