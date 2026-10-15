#!/usr/bin/env python3
"""T1b fix path: onnxsim (fold BN) → onnx2ncnn → patch SE axes -1 → verify vs ONNX."""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import ncnn
import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify

ROOT = Path(__file__).resolve().parents[3]
STEM = "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64"
ONNX = ROOT / "results" / "h618_onnx_v4" / f"{STEM}.onnx"
OUT = ROOT / "results" / "t527_v4" / "t1b_sim_ncnn"
WSL_TOOLS = "<home>/opt/ncnn-x86-tools/bin"
BIN_IN = next((ROOT / "results" / "t527_v4" / "eval500_bins" / "hw64").glob("*.bin"))


def wsl_path(p: Path) -> str:
    s = str(p.resolve())
    return f"/mnt/{s[0].lower()}{s[2:].replace(chr(92), '/')}"


def run_wsl(args: list[str]) -> None:
    if sys.platform.startswith("linux"):
        r = subprocess.run(args, capture_output=True, text=True)
    else:
        cmd = ["wsl", "-e", "bash", "-lc", " ".join(shlex.quote(a) for a in args)]
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout or "") + (r.stderr or ""))


def patch_nan_axes(bin_path: Path) -> int:
    a = np.fromfile(bin_path, dtype=np.float32).copy()
    n = 0
    for i in range(min(64, a.size)):
        if np.isnan(a[i]):
            a[i] = -1.0
            n += 1
    a.tofile(bin_path)
    return n


def ncnn_logits(param: Path, binp: Path, rgb: np.ndarray) -> np.ndarray:
    net = ncnn.Net()
    net.opt.num_threads = 1
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    assert net.load_param(str(param)) == 0
    assert net.load_model(str(binp)) == 0
    h, w = rgb.shape[:2]
    mat = ncnn.Mat.from_pixels(np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB, w, h)
    mat.substract_mean_normalize([0, 0, 0], [1 / 255] * 3)
    ex = net.create_extractor()
    assert ex.input("input", mat) == 0
    ret, out = ex.extract("output")
    assert ret == 0
    return np.array([out[i] for i in range(out.w)], dtype=np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("simplify ONNX (fold BN)...")
    model = onnx.load(str(ONNX))
    sim, ok = simplify(model, skip_fuse_bn=False)
    if not ok:
        print("WARN onnxsim reported not OK, continuing")
    sim_path = OUT / f"{STEM}_sim.onnx"
    onnx.save(sim, str(sim_path))
    print(f"wrote {sim_path}")

    param = OUT / "model.param"
    binp = OUT / "model.bin"
    print("onnx2ncnn...")
    run_wsl([f"{WSL_TOOLS}/onnx2ncnn", wsl_path(sim_path), wsl_path(param), wsl_path(binp)])
    n = patch_nan_axes(binp)
    print(f"patched leading NaN axes -> -1.0 count={n}")
    a = np.fromfile(binp, dtype=np.float32)
    print(f"bin nan={np.isnan(a).sum()} inf={np.isinf(a).sum()} size={a.size}")

    chw = np.fromfile(BIN_IN, dtype=np.float32).reshape(3, 64, 64)
    rgb = (np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    z = ncnn_logits(param, binp, rgb)
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    zo = np.asarray(
        sess.run(None, {sess.get_inputs()[0].name: chw[None]})[0], dtype=np.float32
    ).reshape(-1)
    finite = bool(np.isfinite(z).all())
    cos = (
        float(np.dot(z, zo) / (np.linalg.norm(z) * np.linalg.norm(zo) + 1e-12))
        if finite
        else float("nan")
    )
    print(f"finite={finite} cos={cos:.6f}")
    print("ncnn", np.round(z, 4))
    print("onnx", np.round(zo, 4))
    print("PASS" if finite and cos > 0.999 else "FAIL")
    return 0 if finite and cos > 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
