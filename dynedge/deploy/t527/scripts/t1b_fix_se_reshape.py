#!/usr/bin/env python3
"""Fix onnx2ncnn SE layout: ExpandDims(64,1,1) → Reshape(1,1,C)."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import ncnn
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results/t527_v4/t1b_sim_ncnn"
OUT = ROOT / "results/t527_v4/t1b_sim_fixed"
ONNX = ROOT / "results/h618_onnx_v4/subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64.onnx"
BIN_IN = next((ROOT / "results/t527_v4/eval500_bins/hw64").glob("*.bin"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    param_in = (SRC / "model.param").read_text(encoding="utf-8").splitlines()
    # Drop MemoryData _v_1093 and its Split; rewrite ExpandDims → Reshape
    out_lines: list[str] = []
    skip_names = {"_v_1093", "splitncnn_0"}
    # first two lines magic
    out_lines.append(param_in[0])
    # recount later
    layer_lines: list[str] = []
    blob_names: set[str] = {"input"}
    for line in param_in[2:]:
        parts = line.split()
        if not parts:
            continue
        ltype, lname = parts[0], parts[1]
        if lname in skip_names or ltype == "MemoryData":
            continue
        if ltype == "Split" and parts[3:].count("_v_1093") or (
            ltype == "Split" and any(p.startswith("_v_1093") for p in parts)
        ):
            continue
        if ltype == "ExpandDims":
            # ExpandDims name bottoms... top ...
            # bottoms: data, axes_blob → keep only data
            nb, nt = int(parts[2]), int(parts[3])
            bottoms = parts[4 : 4 + nb]
            tops = parts[4 + nb : 4 + nb + nt]
            data = bottoms[0]
            top = tops[0]
            # infer C from preceding HardSigmoid / Gemm out channels via name heuristics:
            # look at previous kept layer producing HardSigmoid for same stage
            # Default: parse from BinaryOp later; use 64 as placeholder then fix by stage
            # Better: read C from InnerProduct before HardSigmoid in same stage.
            c = 64
            m = re.search(r"stages\.(\d+)\.(\d+)", lname)
            # find most recent InnerProduct se Gemm_1 for this stage in layer_lines
            for prev in reversed(layer_lines):
                pp = prev.split()
                if pp[0] == "InnerProduct" and "se/Gemm_1" in pp[1] and m and m.group(0) in pp[1]:
                    # 0=num_output
                    for tok in pp:
                        if tok.startswith("0="):
                            c = int(tok.split("=")[1])
                            break
                    break
            new = f"Reshape                  {lname} 1 1 {data} {top} 0=1 1=1 2={c}"
            layer_lines.append(new)
            blob_names.add(top)
            continue
        # remove bottoms that reference split _v_1093
        if any("_v_1093" in p for p in parts):
            # skip layers still depending on axes split
            nb, nt = int(parts[2]), int(parts[3])
            bottoms = parts[4 : 4 + nb]
            if any(b.startswith("_v_1093") for b in bottoms):
                continue
        layer_lines.append(line)

    # recount blobs roughly: unique tops
    tops: set[str] = {"input"}
    for line in layer_lines:
        parts = line.split()
        nb, nt = int(parts[2]), int(parts[3])
        for t in parts[4 + nb : 4 + nb + nt]:
            tops.add(t)
    out_lines.append(f"{len(layer_lines)} {len(tops)}")
    out_lines.extend(layer_lines)
    param_out = OUT / "model.param"
    param_out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    # bin: drop first 2 floats (MemoryData axes) from original sim bin
    a = np.fromfile(SRC / "model.bin", dtype=np.float32)
    # original MemoryData w=2 at start
    a2 = a[2:].copy()
    a2.tofile(OUT / "model.bin")
    print(f"wrote {param_out} layers={len(layer_lines)} blobs={len(tops)}")
    print(f"bin {a.size} -> {a2.size}")

    # infer
    chw = np.fromfile(BIN_IN, dtype=np.float32).reshape(3, 64, 64)
    rgb = (np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    net = ncnn.Net()
    net.opt.num_threads = 1
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    if net.load_param(str(param_out)) != 0:
        print("FAIL load_param")
        return 1
    if net.load_model(str(OUT / "model.bin")) != 0:
        print("FAIL load_model")
        return 1
    mat = ncnn.Mat.from_pixels(np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB, 64, 64)
    mat.substract_mean_normalize([0, 0, 0], [1 / 255] * 3)
    ex = net.create_extractor()
    if ex.input("input", mat) != 0:
        print("FAIL input")
        return 1
    ret, out = ex.extract("output")
    if ret != 0:
        print("FAIL extract", ret)
        return 1
    z = np.array([out[i] for i in range(out.w)], dtype=np.float32)
    zo = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    zo = np.asarray(
        zo.run(None, {zo.get_inputs()[0].name: chw[None]})[0], dtype=np.float32
    ).reshape(-1)
    cos = float(np.dot(z, zo) / (np.linalg.norm(z) * np.linalg.norm(zo) + 1e-12))
    print(f"finite={np.isfinite(z).all()} cos={cos:.6f}")
    print("ncnn", np.round(z, 4))
    print("onnx", np.round(zo, 4))
    # check SE mul shape
    net.opt.lightmode = False
    net2 = ncnn.Net()
    net2.opt.lightmode = False
    net2.opt.num_threads = 1
    for k in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic", "use_bf16_storage"):
        setattr(net2.opt, k, False)
    assert net2.load_param(str(param_out)) == 0
    assert net2.load_model(str(OUT / "model.bin")) == 0
    ex2 = net2.create_extractor()
    assert ex2.input("input", mat) == 0
    for name in (
        "/net/stages.0.0/se/Unsqueeze_1_output_0",
        "/net/stages.0.0/se/Mul_output_0",
    ):
        r, o = ex2.extract(name)
        print(name, "ret", r, "shape", (o.w, o.h, o.c) if r == 0 else None)
    print("PASS" if np.isfinite(z).all() and cos > 0.999 else "FAIL")
    return 0 if np.isfinite(z).all() and cos > 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
