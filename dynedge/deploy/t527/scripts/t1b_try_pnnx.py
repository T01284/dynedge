#!/usr/bin/env python3
"""T1b: pnnx convert MCU subnet (fp16=False) and check vs ONNX."""
from __future__ import annotations

import sys
from pathlib import Path

import ncnn
import numpy as np
import onnxruntime as ort
import pnnx
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402
from tools.export_subnet_onnx import StaticSubnet, remap_v1_stem_keys  # noqa: E402

CKPT = ROOT / "results/from_autodl/subnets_v4_predistill/subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_best.pt"
ONNX = ROOT / "results/h618_onnx_v4/subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64.onnx"
OUT = ROOT / "results/t527_v4/t1b_pnnx"
BIN_IN = next((ROOT / "results/t527_v4/eval500_bins/hw64").glob("*.bin"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = remap_v1_stem_keys(ckpt["model"])
    net = ElasticFERSuperNet(num_classes=8, arch="v3_wide", slimmable=True)
    net.load_state_dict(state, strict=True)
    net.eval()
    cfg = SubnetConfig(width=0.35, depths=(2, 2, 2, 2), img_size=64, gray=True)
    wrapped = StaticSubnet(net, cfg).eval()

    ptpath = str(OUT / "model.pt")
    ncnnparam = str(OUT / "model.ncnn.param")
    ncnnbin = str(OUT / "model.ncnn.bin")
    print("pnnx.export fp16=False ...")
    pnnx.export(
        wrapped,
        ptpath,
        inputs=torch.randn(1, 3, 64, 64),
        ncnnparam=ncnnparam,
        ncnnbin=ncnnbin,
        fp16=False,
    )
    print("files", sorted(p.name for p in OUT.glob("*")))

    chw = np.fromfile(BIN_IN, dtype=np.float32).reshape(3, 64, 64)
    rgb = (np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    n = ncnn.Net()
    n.opt.num_threads = 1
    n.opt.use_fp16_packed = False
    n.opt.use_fp16_storage = False
    n.opt.use_fp16_arithmetic = False
    n.opt.use_bf16_storage = False
    assert n.load_param(ncnnparam) == 0
    assert n.load_model(ncnnbin) == 0

    # parse Input/output names
    text = Path(ncnnparam).read_text(encoding="utf-8")
    print(text[:800])
    in_name = "in0"
    for line in text.splitlines():
        if line.startswith("Input"):
            parts = line.split()
            in_name = parts[-1]
            break
    # last top of last layer often out0
    out_name = "out0"
    for line in reversed(text.splitlines()):
        parts = line.split()
        if len(parts) >= 4 and parts[0] not in {"7767517", "Input"}:
            try:
                nb, nt = int(parts[2]), int(parts[3])
                tops = parts[4 + nb : 4 + nb + nt]
                if tops:
                    out_name = tops[0]
                    break
            except Exception:
                continue

    mat = ncnn.Mat.from_pixels(np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB, 64, 64)
    mat.substract_mean_normalize([0, 0, 0], [1 / 255] * 3)
    ex = n.create_extractor()
    assert ex.input(in_name, mat) == 0
    ret, out = ex.extract(out_name)
    if ret != 0:
        # try alternatives
        for cand in ("out0", "output", "logits"):
            ret, out = ex.extract(cand)
            if ret == 0:
                out_name = cand
                break
    assert ret == 0, f"extract failed in={in_name}"
    z = np.array([out[i] for i in range(out.w * out.h * out.c)], dtype=np.float32)
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    zo = np.asarray(
        sess.run(None, {sess.get_inputs()[0].name: chw[None]})[0], dtype=np.float32
    ).reshape(-1)
    # also torch
    with torch.no_grad():
        zt = wrapped(torch.from_numpy(chw[None])).numpy().reshape(-1)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"in={in_name} out={out_name}")
    print(f"ncnn finite={np.isfinite(z).all()} vs_onnx={cos(z,zo):.6f} vs_torch={cos(z,zt):.6f}")
    print("ncnn ", np.round(z, 4))
    print("onnx ", np.round(zo, 4))
    print("torch", np.round(zt, 4))
    ok = np.isfinite(z).all() and cos(z, zo) > 0.999
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
