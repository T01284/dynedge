#!/usr/bin/env python3
"""Diagnose baked pnnx stem0 vs torch."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import ncnn
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402
from tools.export_subnet_onnx import remap_v1_stem_keys  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "bake", ROOT / "deploy/t527/scripts/t1b_bake_static_ncnn.py"
)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)

ckpt = torch.load(m.DEFAULT_CKPT, map_location="cpu", weights_only=False)
slim = ElasticFERSuperNet(num_classes=8, arch="v3_wide", slimmable=True)
slim.load_state_dict(remap_v1_stem_keys(ckpt["model"]), strict=True)
slim.eval()
cfg = SubnetConfig(width=0.35, depths=(2, 2, 2, 2), img_size=64, gray=True)
baked = m.BakedFER(slim, cfg).eval()

OUT = ROOT / "results/t527_v4/t1b_baked_pnnx"
chw = np.fromfile(
    next((ROOT / "results/t527_v4/eval500_bins/hw64").glob("*.bin")), dtype=np.float32
).reshape(3, 64, 64)

with torch.no_grad():
    stem0 = baked.stem[0]
    y = stem0.act(stem0.bn(stem0.conv(torch.from_numpy(chw[None])))).numpy()[0]
    zb = baked(torch.from_numpy(chw[None])).numpy().reshape(-1)

param = OUT / "model.ncnn.param"
binp = OUT / "model.ncnn.bin"
net = ncnn.Net()
net.opt.num_threads = 1
for k in (
    "use_fp16_packed",
    "use_fp16_storage",
    "use_fp16_arithmetic",
    "use_bf16_storage",
):
    setattr(net.opt, k, False)
assert net.load_param(str(param)) == 0 and net.load_model(str(binp)) == 0

rgb = (np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
mat = ncnn.Mat.from_pixels(np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB, 64, 64)
mat.substract_mean_normalize([0, 0, 0], [1 / 255.0] * 3)
xin = np.array(mat).astype(np.float32)
print("input cos", m.cos(xin, chw), "absmax", float(np.abs(xin).max()))

ex = net.create_extractor()
assert ex.input("in0", mat) == 0
ret, o1 = ex.extract("1")
ret2, o2 = ex.extract("2")
assert ret == 0 and ret2 == 0
a1 = np.array(o1).astype(np.float32)
a2 = np.array(o2).astype(np.float32)
print("blob1 absmax", float(np.abs(a1).max()), "cos vs stem0", m.cos(a1, y))
print("blob2 absmax", float(np.abs(a2).max()), "cos vs stem0", m.cos(a2, y))

ret, out = ex.extract("out0")
assert ret == 0
zn = np.array([out[i] for i in range(out.w)], dtype=np.float32)
print("logits cos", m.cos(zn, zb))
print("ncnn", np.round(zn, 4))
print("bake", np.round(zb, 4))
