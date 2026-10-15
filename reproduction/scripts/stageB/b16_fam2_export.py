#!/usr/bin/env python3
"""B16 (item 11): a second architecture family for the proxy-ranking question.

ResNet-style elastic space with random weights (latency does not depend on weights):
3x3 conv stem (stride 2), four stages of BasicBlocks (two dense 3x3 convs, ReLU, identity
or 1x1 projection shortcut), base channels 64/128/256/512, stage strides 1/2/2/2, global
pool + FC(8). No depthwise convs, no SE, no Hardswish -- unlike the MobileNetV3-style
supernet everywhere else in the paper.

Space: width {0.25, 0.5, 0.75, 1.0} x per-stage depth {1, 2, 3}^4 x input {64, 96, 112}.
Sample: 12 width x resolution strata x 8 depth tuples (seed 2026) = 96 subnets.

Per subnet: features (MACs, params, conv/fc layers), ONNX (for Orin TensorRT), NCNN FP32
via pnnx (checked against PyTorch, cos >= 0.999) and NCNN INT8 (ncnn2table with ImageNet
mean/norm on real calibration images, then ncnn2int8).

Outputs: results/stageB/fam2/{sample.json, features.csv, onnx/*.onnx}, deploy/ncnn_edge/fam2/*.param|bin

  python scripts/stageB/b16_fam2_export.py [--skip-ncnn] [--only KEY]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/stageB/fam2"
NCNN_OUT = ROOT / "deploy/ncnn_edge/fam2"
CALIB = ROOT / "data/calib/linux112/calib"
WIDTHS = (0.25, 0.5, 0.75, 1.0)
RES = (64, 96, 112)
DEPTHS = list(itertools.product((1, 2, 3), repeat=4))
BASE = (64, 128, 256, 512)
STRIDES = (1, 2, 2, 2)


def ch(c: int, w: float) -> int:
    return max(8, int(round(c * w / 8)) * 8)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.relu(y + (x if self.short is None else self.short(x)))


class ResNetSubnet(nn.Module):
    def __init__(self, width: float, depths: tuple[int, ...], num_classes: int = 8):
        super().__init__()
        c0 = ch(BASE[0], width)
        self.stem = nn.Sequential(nn.Conv2d(3, c0, 3, 2, 1, bias=False), nn.BatchNorm2d(c0), nn.ReLU(inplace=True))
        blocks, cin = [], c0
        for base, s, d in zip(BASE, STRIDES, depths):
            cout = ch(base, width)
            for i in range(d):
                blocks.append(Block(cin, cout, s if i == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.blocks(self.stem(x))), 1))


def key_of(w: float, d, img: int) -> str:
    return f"res_w{w:g}_d{''.join(map(str, d))}_s{img}c"


def parse(key: str):
    _, w, d, s = key.split("_")
    return float(w[1:]), tuple(int(c) for c in d[1:]), int(s[1:-1])


def build(key: str) -> tuple[nn.Module, int]:
    w, d, img = parse(key)
    torch.manual_seed(sum(map(ord, key)))
    net = ResNetSubnet(w, d)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0.5, 1.0)
            nn.init.uniform_(m.running_var, 0.5, 2.0)
            nn.init.normal_(m.running_mean, 0.0, 0.1)
    return net.eval(), img


def features(net: nn.Module, img: int) -> dict:
    macs, layers = [0], [0]

    def hook(m, x, y):
        if isinstance(m, nn.Conv2d):
            macs[0] += y.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        else:
            macs[0] += m.in_features * m.out_features
        layers[0] += 1

    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        net(torch.zeros(1, 3, img, img))
    for h in hs:
        h.remove()
    return {"macs": macs[0], "params": sum(p.numel() for p in net.parameters()), "n_layers": layers[0]}


def load_prep():
    spec = importlib.util.spec_from_file_location("prep_baked", ROOT / "deploy/t527/scripts/prep_v4_ncnn_baked_pnnx.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["prep_baked"] = m
    spec.loader.exec_module(m)
    return m


def export_ncnn(prep, key: str, net: nn.Module, img: int, imagelist: Path) -> float:
    import ncnn
    import pnnx

    param, binp = NCNN_OUT / f"{key}.param", NCNN_OUT / f"{key}.bin"
    iparam = NCNN_OUT / f"{key}_int8.param"
    with tempfile.TemporaryDirectory(dir=ROOT / "results/stageB/fam2") as tmp:
        t = Path(tmp)
        x = torch.randn(1, 3, img, img)
        try:
            pnnx.export(net, str(t / "m.pt"), inputs=x, ncnnparam=str(t / "m.ncnn.param"),
                        ncnnbin=str(t / "m.ncnn.bin"), fp16=False)
        except Exception as e:  # pnnx raises after writing files when it cannot import model_pnnx.py
            print(f"  pnnx post-err (ok if files exist): {type(e).__name__}")
        prep.rename_io_blobs(t / "m.ncnn.param")
        param.write_text((t / "m.ncnn.param").read_text(encoding="utf-8"), encoding="utf-8")
        binp.write_bytes((t / "m.ncnn.bin").read_bytes())
    nt = ncnn.Net()
    nt.opt.num_threads = 1
    nt.opt.use_fp16_packed = nt.opt.use_fp16_storage = nt.opt.use_fp16_arithmetic = False
    nt.opt.use_bf16_storage = False
    assert nt.load_param(str(param)) == 0 and nt.load_model(str(binp)) == 0
    ex = nt.create_extractor()
    ex.input("input", ncnn.Mat(x[0].numpy()))
    _, out = ex.extract("output")
    zn = np.array([out[i] for i in range(out.w)], dtype=np.float32)
    with torch.no_grad():
        zt = net(x).numpy().reshape(-1)
    c = float(np.dot(zn, zt) / (np.linalg.norm(zn) * np.linalg.norm(zt) + 1e-12))
    if c < 0.999:
        raise RuntimeError(f"{key}: ncnn vs torch cos={c}")
    if not iparam.exists():
        table = NCNN_OUT / f"{key}.table"
        r = prep.run_tools([f"{prep.WSL_TOOLS}/ncnn2table", prep.wsl_path(param), prep.wsl_path(binp),
                            prep.wsl_path(imagelist), prep.wsl_path(table),
                            f"mean=[{','.join(f'{v:g}' for v in prep.IMAGENET_MEAN_255)}]",
                            f"norm=[{','.join(f'{v:.8f}' for v in prep.IMAGENET_NORM_255)}]",
                            f"shape=[{img},{img},3]", "pixel=RGB", "thread=8"])
        if r.returncode != 0 or not table.exists():
            raise RuntimeError(f"ncnn2table {key}: {(r.stderr or r.stdout)[-500:]}")
        r = prep.run_tools([f"{prep.WSL_TOOLS}/ncnn2int8", prep.wsl_path(param), prep.wsl_path(binp),
                            prep.wsl_path(iparam), prep.wsl_path(NCNN_OUT / f"{key}_int8.bin"),
                            prep.wsl_path(table)])
        if r.returncode != 0 or not iparam.exists():
            raise RuntimeError(f"ncnn2int8 {key}: {(r.stderr or r.stdout)[-500:]}")
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip-ncnn", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    keys = [key_of(w, d, img) for w in WIDTHS for img in RES for d in rng.sample(DEPTHS, args.per_stratum)]
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    NCNN_OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps({"seed": args.seed, "per_stratum": args.per_stratum,
                                                 "n": len(keys), "keys": keys}, indent=2), encoding="utf-8")
    prep = None if args.skip_ncnn else load_prep()
    imagelist = None
    if prep:
        imagelist = prep.ensure_imagelist(CALIB, NCNN_OUT / "imagelist_calib.txt", limit=100)

    rows = []
    for i, key in enumerate(keys):
        net, img = build(key)
        f = features(net, img)
        onnx = OUT / "onnx" / f"{key}.onnx"
        if not onnx.exists():
            with torch.no_grad():
                torch.onnx.export(net, torch.randn(1, 3, img, img), str(onnx), input_names=["input"],
                                  output_names=["logits"], opset_version=17, dynamo=False)
        c = float("nan")
        if prep and (not args.only or args.only in key):
            c = export_ncnn(prep, key, net, img, imagelist)
        rows.append({"arch": key, "img": img, **f, "ncnn_cos": c})
        print(f"{i + 1}/{len(keys)} {key} MMACs={f['macs'] / 1e6:.1f} params={f['params'] / 1e6:.2f}M "
              f"layers={f['n_layers']} cos={c:.6f}", flush=True)
    with (OUT / "features.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT / 'features.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
