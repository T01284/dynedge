#!/usr/bin/env python3
"""T1b: find first NaN layer in NCNN FP32 graph (host pyncnn)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import ncnn
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STEM = "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64"
MODEL = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
BIN = next((ROOT / "results" / "t527_v4" / "eval500_bins" / "hw64").glob("*.bin"))
OUT = ROOT / "results" / "t527_v4" / "t1b_first_nan.json"


def md5(p: Path) -> str:
    h = hashlib.md5()
    h.update(p.read_bytes())
    return h.hexdigest()


def parse_blobs(param: Path) -> list[str]:
    """Collect output blob names in topo order from ncnn param."""
    lines = param.read_text(encoding="utf-8", errors="replace").splitlines()
    # skip magic + layer count
    blobs: list[str] = []
    seen = set()
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        # type name bottom_count top_count bottoms... tops...
        if len(parts) < 4:
            continue
        try:
            n_bottom = int(parts[2])
            n_top = int(parts[3])
        except ValueError:
            continue
        tops = parts[4 + n_bottom : 4 + n_bottom + n_top]
        for t in tops:
            if t not in seen and t != "input":
                seen.add(t)
                blobs.append(t)
    return blobs


def main() -> int:
    param = MODEL / f"{STEM}.param"
    binp = MODEL / f"{STEM}.bin"
    info = {
        "stem": STEM,
        "param_md5": md5(param),
        "bin_md5": md5(binp),
        "input_bin": str(BIN.relative_to(ROOT)).replace("\\", "/"),
    }
    chw = np.fromfile(BIN, dtype=np.float32).reshape(3, 64, 64)
    rgb = (np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)

    net = ncnn.Net()
    net.opt.num_threads = 1
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    assert net.load_param(str(param)) == 0
    assert net.load_model(str(binp)) == 0

    blobs = parse_blobs(param)
    print(f"blobs={len(blobs)} param_md5={info['param_md5']} bin_md5={info['bin_md5']}")

    first = None
    rows = []
    # Recreate extractor each time? Better one pass extracting sequentially —
    # ncnn allows extract of intermediate blobs if lightmode may recycle.
    # Disable lightmode to keep intermediates.
    net.opt.lightmode = False
    # reload after option change
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
    mat.substract_mean_normalize([0, 0, 0], [1 / 255, 1 / 255, 1 / 255])
    ex = net.create_extractor()
    assert ex.input("input", mat) == 0

    for i, name in enumerate(blobs):
        ret, out = ex.extract(name)
        if ret != 0:
            rows.append({"i": i, "blob": name, "status": f"extract_fail_{ret}"})
            print(f"[{i}] FAIL extract {name} ret={ret}")
            if first is None:
                first = {"i": i, "blob": name, "reason": f"extract_fail_{ret}"}
            break
        arr = np.array(out).astype(np.float32).reshape(-1)
        finite = bool(np.isfinite(arr).all())
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        amax = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
        rows.append(
            {
                "i": i,
                "blob": name,
                "status": "ok" if finite else "nonfinite",
                "n": int(arr.size),
                "n_nan": n_nan,
                "n_inf": n_inf,
                "absmax": amax,
            }
        )
        tag = "OK" if finite else "NAN"
        print(f"[{i:03d}] {tag} absmax={amax:.4g} nan={n_nan} inf={n_inf} {name}")
        if not finite and first is None:
            first = {
                "i": i,
                "blob": name,
                "n_nan": n_nan,
                "n_inf": n_inf,
                "absmax": amax,
            }
            # continue a few more to see contamination
            if i > first["i"] + 5:
                break

    info["first_nonfinite"] = first
    info["layers"] = rows
    OUT.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"FIRST={first}")
    print(f"wrote {OUT}")
    return 0 if first else 1


if __name__ == "__main__":
    raise SystemExit(main())
