#!/usr/bin/env python3
"""T527 v4 anchor: NPU uint8 NBG vs NCNN FP32 top-1 match.

Requires *_nbg_meta.json. Uses ImageNet-BGR preprocess (matches ACUITY inputmeta).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image

from ssh_t527 import put, run

ROOT = Path(__file__).resolve().parents[3]
NBG_DIR = ROOT / "deploy" / "t527" / "models" / "v4_npu"
PARAM_DIR = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
EVAL112 = ROOT / "data" / "calib" / "linux112" / "eval" / "ferplus"
EVAL64 = ROOT / "data" / "calib" / "ferplus_mcu64" / "test"
OUT = ROOT / "results" / "t527_v4" / "npu_vs_ncnn_acc.json"
REMOTE_BASE = "/root/fer-dynedge/npu/v4_acc"

MEAN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
SCALE = np.array([1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0], dtype=np.float32)


def hw_of(stem: str) -> int:
    if "img64" in stem or "_s64g_" in stem:
        return 64
    if "img96" in stem or "_s96c_" in stem:
        return 96
    return 112


def short(stem: str) -> str:
    import re

    m = re.search(r"w[\d.]+_d\d+_s\d+[gc]", stem)
    return (m.group(0).replace(".", "") if m else "x") + "_acc"


def list_imgs(root: Path, limit: int) -> list[Path]:
    """Round-robin across class folders for less biased match estimate."""
    by_cls: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        by_cls.setdefault(p.parent.name, []).append(p)
    keys = sorted(by_cls)
    out: list[Path] = []
    i = 0
    while len(out) < limit and keys:
        progressed = False
        for k in keys:
            if i < len(by_cls[k]):
                out.append(by_cls[k][i])
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
        i += 1
    return out


def float_imagenet_nchw(rgb: np.ndarray) -> np.ndarray:
    """RGB /255 NCHW (name kept for call sites)."""
    x = rgb.astype(np.float32) * (1.0 / 255.0)
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))


def quantize_u8(chw: np.ndarray, scale: float, zp: int) -> bytes:
    q = np.rint(chw / scale + zp).astype(np.int32)
    return np.clip(q, 0, 255).astype(np.uint8).tobytes(order="C")


def load_ncnn(param: Path, binp: Path) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.num_threads = 2
    if net.load_param(str(param)) != 0:
        raise RuntimeError(f"load_param {param}")
    if net.load_model(str(binp)) != 0:
        raise RuntimeError(f"load_model {binp}")
    return net


def ncnn_argmax(net: ncnn.Net, chw: np.ndarray) -> int:
    mat = ncnn.Mat(np.ascontiguousarray(chw))
    ex = net.create_extractor()
    if ex.input("input", mat) != 0:
        raise RuntimeError("input failed")
    ret, out = ex.extract("output")
    if ret != 0:
        raise RuntimeError(f"extract {ret}")
    scores = np.array([out[i] for i in range(out.w * out.h * out.c)], dtype=np.float32)
    return int(np.argmax(scores))


def sample_txt(remote: str, dat: str) -> str:
    return (
        "vpm_run is a framework for testing VIPLite driver. \n"
        "\n"
        "#You may put comment by starting with a '#'.\n"
        "#####################################################A separator makes the list clear.\n"
        "[network]\n"
        f"{remote}/network_binary.nb\n"
        "[input]\n"
        f"{remote}/{dat}\n"
        "\n"
    )


def npu_argmax(remote: str, dat_bytes: bytes, hw: int) -> int:
    dat = f"{hw}x{hw}_u8.dat"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / dat).write_bytes(dat_bytes)
        (p / "sample.txt").write_text(sample_txt(remote, dat), encoding="utf-8", newline="\n")
        put(str(p / dat), f"{remote}/{dat}")
        put(str(p / "sample.txt"), f"{remote}/sample.txt")
    code, out, err = run(
        f"chmod 666 /dev/vipcore 2>/dev/null || true; "
        f"cd {remote} && rm -f output_0.txt && "
        f"vpm_run -s sample.txt -l 1 -b 0 --save_txt 1 2>&1",
        timeout=180,
    )
    if code != 0:
        raise RuntimeError((out + err)[-1500:])
    _, txt, _ = run(f"cat {remote}/output_0.txt")
    vals = [float(x.strip()) for x in txt.splitlines() if x.strip()]
    if len(vals) < 2:
        raise RuntimeError(f"bad output_0.txt: {txt[:300]}")
    return int(np.argmax(np.asarray(vals, dtype=np.float32)))


def eval_one(stem: str, limit: int) -> dict:
    nbg = NBG_DIR / f"{stem}_uint8.nb"
    meta_p = NBG_DIR / f"{stem}_nbg_meta.json"
    param = PARAM_DIR / f"{stem}.param"
    binp = PARAM_DIR / f"{stem}.bin"
    if not meta_p.exists():
        return {"model": stem, "status": "skip_no_meta"}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    q = next(iter(meta["Inputs"].values()))["quantize"]
    scale, zp = float(q["scale"]), int(q["zero_point"])
    hw = hw_of(stem)
    imgs = list_imgs(EVAL64 if hw == 64 else EVAL112, limit)
    if not imgs:
        return {"model": stem, "status": "skip_no_images"}

    remote = f"{REMOTE_BASE}/{short(stem)}"
    run(f"mkdir -p {remote}")
    put(str(nbg), f"{remote}/network_binary.nb")
    net = load_ncnn(param, binp)

    match = 0
    for i, img in enumerate(imgs):
        rgb = np.asarray(
            Image.open(img).convert("RGB").resize((hw, hw), Image.BILINEAR),
            dtype=np.uint8,
        )
        if hw == 64:
            # training used --gray: replicate luma to 3ch
            g = np.asarray(
                Image.open(img).convert("L").resize((hw, hw), Image.BILINEAR),
                dtype=np.uint8,
            )
            rgb = np.stack([g, g, g], axis=-1)
        chw = float_imagenet_nchw(rgb)
        pred_n = npu_argmax(remote, quantize_u8(chw, scale, zp), hw)
        pred_c = ncnn_argmax(net, chw)
        if pred_n == pred_c:
            match += 1
        if (i + 1) % 10 == 0:
            print(f"  {stem[-40:]} {i+1}/{len(imgs)} match={match/(i+1):.3f}", flush=True)

    return {
        "model": stem,
        "status": "ok",
        "n": len(imgs),
        "match": match,
        "match_rate": round(match / len(imgs), 4),
        "input_scale": scale,
        "input_zp": zp,
        "hw": hw,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    stems = sorted(p.name.replace("_nbg_meta.json", "") for p in NBG_DIR.glob("*_nbg_meta.json"))
    if args.only:
        stems = [s for s in stems if args.only in s]
    results = []
    for stem in stems:
        print(f"== {stem} ==", flush=True)
        try:
            r = eval_one(stem, args.limit)
        except Exception as ex:
            r = {"model": stem, "status": f"fail:{ex}"}
        print(r, flush=True)
        results.append(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"DONE -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
