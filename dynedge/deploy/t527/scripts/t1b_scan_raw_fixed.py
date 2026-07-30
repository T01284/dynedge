#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import ncnn
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
raw = ROOT / "results" / "t527_v4" / "t1b_raw_ncnn"
param = raw / "raw_fixed.param"
binp = raw / "raw_fixed.bin"
chw = np.fromfile(
    next((ROOT / "results" / "t527_v4" / "eval500_bins" / "hw64").glob("*.bin")),
    dtype=np.float32,
).reshape(3, 64, 64)
rgb = (np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)


def parse_blobs(p: Path) -> list[str]:
    lines = p.read_text(encoding="utf-8").splitlines()
    blobs: list[str] = []
    seen: set[str] = set()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            nb, nt = int(parts[2]), int(parts[3])
        except ValueError:
            continue
        tops = parts[4 + nb : 4 + nb + nt]
        for t in tops:
            if t not in seen and t != "input":
                seen.add(t)
                blobs.append(t)
    return blobs


def main() -> int:
    net = ncnn.Net()
    net.opt.num_threads = 1
    net.opt.lightmode = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    assert net.load_param(str(param)) == 0
    assert net.load_model(str(binp)) == 0
    mat = ncnn.Mat.from_pixels(np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB, 64, 64)
    mat.substract_mean_normalize([0, 0, 0], [1 / 255] * 3)
    ex = net.create_extractor()
    assert ex.input("input", mat) == 0
    first = None
    for i, name in enumerate(parse_blobs(param)):
        ret, out = ex.extract(name)
        if ret != 0:
            print(i, "FAIL", name, ret)
            break
        arr = np.array(out).astype(np.float32).reshape(-1)
        finite = bool(np.isfinite(arr).all())
        amax = float(np.nanmax(np.abs(arr))) if arr.size and not np.isnan(arr).all() else float("nan")
        tag = "OK" if finite else "NAN"
        print(f"[{i:03d}] {tag} absmax={amax:.4g} nan={int(np.isnan(arr).sum())} {name}")
        if not finite and first is None:
            first = (i, name)
    print("FIRST", first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
