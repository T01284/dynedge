#!/usr/bin/env python3
"""T1b/T1c: bake slimmable v4 → pnnx ncnn FP32, then aciq INT8 with matched calib.

Replaces broken onnx2ncnn+optimize path (SE Unsqueeze NaN / wrong logits).
Board runners expect blob names input/output — rewritten after pnnx (in0/out0).
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shlex
import subprocess
import sys
from pathlib import Path

import ncnn
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402
from tools.export_subnet_onnx import remap_v1_stem_keys  # noqa: E402

CKPT_DIR = ROOT / "results/from_autodl/subnets_v4_predistill"
OUT_DIR = ROOT / "deploy/t527/models/v4_predistill"
CALIB112 = ROOT / "data/calib/linux112/calib"
CALIB64 = ROOT / "data/calib/ferplus_mcu64"
WSL_TOOLS = "<home>/opt/ncnn-x86-tools/bin"
BIN_ROOT = ROOT / "results/t527_v4/eval500_bins"

_bake_spec = importlib.util.spec_from_file_location(
    "t1b_bake", ROOT / "deploy/t527/scripts/t1b_bake_static_ncnn.py"
)
_bake = importlib.util.module_from_spec(_bake_spec)
assert _bake_spec.loader is not None
_bake_spec.loader.exec_module(_bake)
BakedFER = _bake.BakedFER
cos = _bake.cos


def wsl_path(p: Path) -> str:
    s = str(p.resolve())
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:].replace(chr(92), '/')}"
    return s


def run_tools(args: list[str]) -> subprocess.CompletedProcess:
    if sys.platform.startswith("linux"):
        return subprocess.run(args, capture_output=True, text=True)
    cmd = ["wsl", "-e", "bash", "-lc", " ".join(shlex.quote(a) for a in args)]
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_stem(stem: str) -> tuple[str, float, tuple[int, int, int, int], int, bool]:
    m = re.search(r"subnet_(ferplus|expw)_w([0-9.]+)_d(\d{4})_s(\d+)([gc])_", stem)
    if not m:
        raise ValueError(f"cannot parse stem {stem}")
    ds, w_s, d_s, res_s, gc = m.groups()
    width = float(w_s)
    depths = tuple(int(d_s[i]) for i in range(4))  # type: ignore[assignment]
    img = int(res_s)
    gray = gc == "g"
    return ds, width, depths, img, gray  # type: ignore[return-value]


def ckpt_for(stem: str, tag: str = "v4_predistill") -> Path:
    ds, width, depths, img, gray = parse_stem(stem)
    # filename uses w0.35 / w0.5 / w1 (no trailing .0)
    if width == int(width):
        wtag = f"w{int(width)}"
    else:
        wtag = f"w{width}"
    dtag = "d" + "".join(str(d) for d in depths)
    stag = f"s{img}{'g' if gray else 'c'}"
    return CKPT_DIR / f"subnet_{ds}_{wtag}_{dtag}_{stag}_seed42_{tag}_best.pt"


def rename_io_blobs(param: Path) -> None:
    text = param.read_text(encoding="utf-8")
    text2 = text.replace(" in0", " input").replace("\tin0", "\tinput")
    # Input layer name column and blob
    lines = []
    for line in text2.splitlines():
        if line.startswith("Input"):
            parts = line.split()
            # Input name 0 1 blob
            if len(parts) >= 4:
                parts[1] = "input"
                parts[3] = "input"
                line = " ".join(parts) if "\t" not in line else line  # keep spaces style
                # pnnx uses spaces with padding — rewrite simply
                line = f"Input                    input                     0 1 input"
        lines.append(line)
    text2 = "\n".join(lines) + "\n"
    # rename final out0 → output (blob references)
    text2 = text2.replace(" out0", " output")
    param.write_text(text2, encoding="utf-8")


def rgb_u8_from_chw(chw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        (np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255)).astype(np.uint8)
    )


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
    mat = ncnn.Mat.from_pixels(
        np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB, w, h
    )
    mat.substract_mean_normalize([0, 0, 0], [1 / 255.0] * 3)
    ex = net.create_extractor()
    assert ex.input("input", mat) == 0
    ret, out = ex.extract("output")
    assert ret == 0
    return np.array([out[i] for i in range(out.w)], dtype=np.float32)


def export_one(stem: str, work: Path, *, ckpt_tag: str = "v4_predistill", relu: bool = False) -> tuple[Path, Path, float]:
    import pnnx

    ds, width, depths, img, gray = parse_stem(stem)
    ckpt_path = ckpt_for(stem, tag=ckpt_tag)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    cfg = SubnetConfig(width=width, depths=depths, img_size=img, gray=gray)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = remap_v1_stem_keys(ckpt["model"])
    num_classes = int(state["final_fc.weight"].shape[0])
    slim = ElasticFERSuperNet(num_classes=num_classes, arch="v3_wide", slimmable=True)
    slim.load_state_dict(state, strict=True)
    if relu:
        from tools.export_subnet_onnx import hardswish_to_relu

        hardswish_to_relu(slim)
    slim.eval()
    baked = BakedFER(slim, cfg).eval()

    work.mkdir(parents=True, exist_ok=True)
    # short path avoids broken escapes in generated model_pnnx.py on Windows
    pt = str(work / "m.pt")
    param = work / "m.ncnn.param"
    binp = work / "m.ncnn.bin"
    xt = torch.randn(1, 3, img, img)
    try:
        pnnx.export(
            baked,
            pt,
            inputs=xt,
            ncnnparam=str(param),
            ncnnbin=str(binp),
            fp16=False,
        )
    except Exception as e:
        print(f"  pnnx.export post-err (ok if files exist): {type(e).__name__}: {e}")
    if not param.exists() or not binp.exists():
        raise RuntimeError(f"pnnx failed to write ncnn for {stem}")
    rename_io_blobs(param)

    bin_dir = BIN_ROOT / f"hw{img}"
    sample = next(bin_dir.glob("*.bin"))
    chw = np.fromfile(sample, dtype=np.float32).reshape(3, img, img)
    with torch.no_grad():
        zb = baked(torch.from_numpy(chw[None])).numpy().reshape(-1)
    zn = ncnn_logits(param, binp, rgb_u8_from_chw(chw))
    c = cos(zn, zb)
    print(f"  verify cos={c:.8f} finite={bool(np.isfinite(zn).all())}")
    if not np.isfinite(zn).all() or c < 0.999:
        raise RuntimeError(f"verify fail {stem} cos={c}")

    out_param = OUT_DIR / f"{stem}.param"
    out_bin = OUT_DIR / f"{stem}.bin"
    out_param.write_bytes(param.read_bytes())
    out_bin.write_bytes(binp.read_bytes())
    # also copy renamed param text
    out_param.write_text(param.read_text(encoding="utf-8"), encoding="utf-8")
    return out_param, out_bin, c


def ensure_imagelist(calib_root: Path, out: Path, limit: int = 0) -> Path:
    files = sorted(
        p
        for p in calib_root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if limit:
        files = files[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(wsl_path(p) for p in files) + "\n", encoding="utf-8")
    return out


def build_int8(stem: str, param: Path, binp: Path, list64: Path, list112: Path) -> None:
    _, _, _, img, _ = parse_stem(stem)
    calib = list64 if img == 64 else list112
    # pnnx mean/norm already in graph? Our graph expects 0-255 then /255 via runner.
    # ncnn2table uses mean/norm flags — match fer_runner: mean 0, norm 1/255
    table = OUT_DIR / f"{stem}.table"
    iparam = OUT_DIR / f"{stem}_int8.param"
    ibin = OUT_DIR / f"{stem}_int8.bin"
    shape = f"[{img},{img},3]"
    r = run_tools(
        [
            f"{WSL_TOOLS}/ncnn2table",
            wsl_path(param),
            wsl_path(binp),
            wsl_path(calib),
            wsl_path(table),
            "mean=[0,0,0]",
            "norm=[0.00392157,0.00392157,0.00392157]",
            f"shape={shape}",
            "pixel=RGB",
            "thread=4",
        ]
    )
    if r.returncode != 0 or not table.exists():
        raise RuntimeError(f"ncnn2table {stem}: {(r.stderr or r.stdout)[-800:]}")
    # clamp non-finite scales in table
    lines = []
    for line in table.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                v = float(parts[-1])
                if not np.isfinite(v) or v <= 0:
                    parts[-1] = "1.0"
            except ValueError:
                pass
        lines.append(" ".join(parts))
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = run_tools(
        [
            f"{WSL_TOOLS}/ncnn2int8",
            wsl_path(param),
            wsl_path(binp),
            wsl_path(iparam),
            wsl_path(ibin),
            wsl_path(table),
        ]
    )
    if r.returncode != 0 or not iparam.exists():
        raise RuntimeError(f"ncnn2int8 {stem}: {(r.stderr or r.stdout)[-800:]}")


def main() -> int:
    global CKPT_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stem", type=str, default="")
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--fp32-only-stems", action="store_true")
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--onnx-dir", type=Path, default=None)
    ap.add_argument("--ckpt-tag", type=str, default="v4_predistill")
    ap.add_argument("--relu", action="store_true")
    args = ap.parse_args()
    if args.ckpt_dir:
        CKPT_DIR = args.ckpt_dir
    if args.out_dir:
        OUT_DIR = args.out_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_dir = args.onnx_dir or (ROOT / "results/h618_onnx_v4")
    stems = sorted(p.stem for p in onnx_dir.glob("*.onnx"))
    if args.stem:
        stems = [args.stem]
    if args.limit:
        stems = stems[: args.limit]

    list112 = ensure_imagelist(CALIB112, OUT_DIR / "imagelist_real.txt")
    list64 = ensure_imagelist(CALIB64, OUT_DIR / "imagelist_mcu64.txt", limit=200)

    work_root = ROOT / "results/t527_v4/baked_pnnx"
    ok = fail = 0
    results = []
    for stem in stems:
        print(f"== {stem} ==", flush=True)
        try:
            param, binp, c = export_one(
                stem, work_root / stem, ckpt_tag=args.ckpt_tag, relu=args.relu
            )
            if not args.skip_int8:
                print("  int8...", flush=True)
                build_int8(stem, param, binp, list64, list112)
            results.append({"stem": stem, "cos": c, "status": "ok"})
            ok += 1
            print(f"OK {stem} cos={c:.6f}", flush=True)
        except Exception as e:
            results.append({"stem": stem, "status": f"fail:{e}"})
            fail += 1
            print(f"FAIL {stem}: {e}", flush=True)

    summary = work_root / "convert_summary.json"
    import json

    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"done ok={ok} fail={fail} -> {summary}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
