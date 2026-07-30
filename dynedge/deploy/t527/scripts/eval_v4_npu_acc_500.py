#!/usr/bin/env python3
"""T2a: T527 v4 NPU uint8 NBG vs NCNN FP32 (bake+pnnx) top-1 match @500.

Replaces voided n≤48 §8.10 numbers. Uses same eval pool protocol as T1-fix when
--pool-list is provided; otherwise stratified sample from ferplus eval roots.

Runtime: FP16 off; contiguous RGB → from_pixels (not Mat(chw) which mis-parses).
"""
from __future__ import annotations

import argparse
import csv
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
EVAL112 = ROOT / "data" / "calib" / "linux112" / "eval"
EVAL64 = ROOT / "data" / "calib" / "ferplus_mcu64"
OUT_JSON = ROOT / "results" / "t527_v4" / "npu_vs_ncnn_acc_500.json"
OUT_CSV = ROOT / "results" / "t527_npu_acc_500.csv"
POOL_LIST = ROOT / "results" / "t527_v4" / "eval500_imagelist.txt"
REMOTE_BASE = "/root/fer-dynedge/npu/v4_acc500"


def hw_of(stem: str) -> int:
    if "img64" in stem or "_s64g_" in stem:
        return 64
    if "img96" in stem or "_s96c_" in stem:
        return 96
    return 112


def short(stem: str) -> str:
    import re

    m = re.search(r"w[\d.]+_d\d+_s\d+[gc]", stem)
    return (m.group(0).replace(".", "") if m else "x") + "_a500"


def list_imgs_stratified(root: Path, limit: int, dataset: str = "ferplus") -> list[Path]:
    by_cls: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        parts = p.relative_to(root).parts
        # linux112/eval/{ferplus|expw}/class/...
        if len(parts) >= 2 and parts[0] in {"ferplus", "expw"}:
            if parts[0] != dataset:
                continue
            cls = parts[1]
        else:
            cls = p.parent.name
        by_cls.setdefault(cls, []).append(p)
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


def load_pool(limit: int, hw: int) -> list[Path]:
    """Prefer T1 eval500 list; for 64g use all mcu64 images (up to limit)."""
    if hw == 64:
        imgs: list[Path] = []
        for cand in (EVAL64 / "test", EVAL64 / "calib", EVAL64):
            if not cand.exists():
                continue
            imgs = list_imgs_stratified(cand if cand != EVAL64 else EVAL64, limit)
            # Prefer the root that actually has many files
            if len(imgs) >= min(limit, 200):
                return imgs[:limit]
        # Fall back: all images under mcu64
        all_imgs = sorted(
            p
            for p in EVAL64.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if all_imgs:
            return all_imgs[:limit]
        return imgs[:limit]
    if POOL_LIST.exists() and hw >= 96:
        paths = []
        for line in POOL_LIST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = ROOT / line if not Path(line).is_absolute() else Path(line)
            if p.exists():
                paths.append(p)
            if len(paths) >= limit:
                break
        if len(paths) >= min(limit, 100):
            return paths[:limit]
    root = EVAL112 / "eval" if (EVAL112 / "eval").exists() else EVAL112
    return list_imgs_stratified(root, limit)


def load_rgb_u8(path: Path, hw: int, gray: bool) -> np.ndarray:
    if gray:
        g = np.asarray(
            Image.open(path).convert("L").resize((hw, hw), Image.BILINEAR),
            dtype=np.uint8,
        )
        return np.ascontiguousarray(np.stack([g, g, g], axis=-1))
    return np.ascontiguousarray(
        np.asarray(
            Image.open(path).convert("RGB").resize((hw, hw), Image.BILINEAR),
            dtype=np.uint8,
        )
    )


def float_nchw(rgb: np.ndarray) -> np.ndarray:
    return np.transpose(rgb.astype(np.float32) * (1.0 / 255.0), (2, 0, 1))


def quantize_u8(chw: np.ndarray, scale: float, zp: int) -> bytes:
    q = np.rint(chw / scale + zp).astype(np.int32)
    return np.clip(q, 0, 255).astype(np.uint8).tobytes(order="C")


def load_ncnn(param: Path, binp: Path) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.num_threads = 2
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    if net.load_param(str(param)) != 0:
        raise RuntimeError(f"load_param {param}")
    if net.load_model(str(binp)) != 0:
        raise RuntimeError(f"load_model {binp}")
    return net


def ncnn_argmax(net: ncnn.Net, rgb_u8: np.ndarray) -> int:
    h, w = rgb_u8.shape[:2]
    mat = ncnn.Mat.from_pixels(
        np.ascontiguousarray(rgb_u8), ncnn.Mat.PixelType.PIXEL_RGB, w, h
    )
    mat.substract_mean_normalize([0.0, 0.0, 0.0], [1.0 / 255.0] * 3)
    ex = net.create_extractor()
    if ex.input("input", mat) != 0:
        raise RuntimeError("input failed")
    ret, out = ex.extract("output")
    if ret != 0:
        raise RuntimeError(f"extract {ret}")
    scores = np.array([out[i] for i in range(out.w * out.h * out.c)], dtype=np.float32)
    if not np.isfinite(scores).all():
        raise RuntimeError("ncnn logits non-finite")
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


NOTE_OVERRIDE = ""


def eval_one(stem: str, limit: int) -> dict:
    nbg = NBG_DIR / f"{stem}_uint8.nb"
    meta_p = NBG_DIR / f"{stem}_nbg_meta.json"
    param = PARAM_DIR / f"{stem}.param"
    binp = PARAM_DIR / f"{stem}.bin"
    for p in (nbg, meta_p, param, binp):
        if not p.exists():
            return {"model": stem, "status": f"missing:{p.name}"}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    q = next(iter(meta["Inputs"].values()))["quantize"]
    scale, zp = float(q["scale"]), int(q["zero_point"])
    hw = hw_of(stem)
    gray = hw == 64
    imgs = load_pool(limit, hw)
    if not imgs:
        return {"model": stem, "status": "skip_no_images"}

    remote = f"{REMOTE_BASE}/{short(stem)}"
    run(f"mkdir -p {remote}")
    put(str(nbg), f"{remote}/network_binary.nb")
    net = load_ncnn(param, binp)

    match = 0
    n_nan = 0
    for i, img in enumerate(imgs):
        rgb = load_rgb_u8(img, hw, gray=gray)
        chw = float_nchw(rgb)
        try:
            pred_c = ncnn_argmax(net, rgb)
            pred_n = npu_argmax(remote, quantize_u8(chw, scale, zp), hw)
        except Exception as ex:
            n_nan += 1
            if n_nan <= 3:
                print(f"  warn img={img.name}: {ex}", flush=True)
            continue
        match += int(pred_n == pred_c)
        n_ok = (i + 1) - n_nan
        if (i + 1) % 25 == 0:
            rate = match / max(n_ok, 1)
            print(f"  {stem[-48:]} {i+1}/{len(imgs)} match={rate:.3f} nan={n_nan}", flush=True)

    n_ok = len(imgs) - n_nan
    if n_ok == 0:
        return {"model": stem, "status": "fail_no_finite", "n_nan": n_nan}
    return {
        "model": stem,
        "status": "ok",
        "n": n_ok,
        "n_nan": n_nan,
        "n_pool": len(imgs),
        "match": match,
        "match_rate": round(match / n_ok, 4),
        "input_scale": scale,
        "input_zp": zp,
        "hw": hw,
        "fp32_ref": "ncnn_baked_pnnx",
        "note": NOTE_OVERRIDE or "t2a_hardswish_nbg",
    }


def main() -> int:
    global NBG_DIR, PARAM_DIR, OUT_CSV, OUT_JSON, REMOTE_BASE, NOTE_OVERRIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--only", default="")
    ap.add_argument("--nbg-dir", type=Path, default=None)
    ap.add_argument("--param-dir", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--remote-base", type=str, default="")
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()
    if args.nbg_dir:
        NBG_DIR = args.nbg_dir
    if args.param_dir:
        PARAM_DIR = args.param_dir
    if args.out_csv:
        OUT_CSV = args.out_csv
    if args.out_json:
        OUT_JSON = args.out_json
    if args.remote_base:
        REMOTE_BASE = args.remote_base
    if args.note:
        NOTE_OVERRIDE = args.note
    stems = sorted(p.name.replace("_nbg_meta.json", "") for p in NBG_DIR.glob("*_nbg_meta.json"))
    if args.only:
        stems = [s for s in stems if args.only in s]

    # smoke board
    code, out, err = run("uname -a; which vpm_run; ls /dev/vipcore", timeout=30)
    print(out or err, flush=True)
    if code != 0:
        print("ERR board not ready", flush=True)
        return 1

    results = []
    for stem in stems:
        print(f"== {stem} ==", flush=True)
        try:
            r = eval_one(stem, args.limit)
        except Exception as ex:
            r = {"model": stem, "status": f"fail:{ex}"}
        print(r, flush=True)
        results.append(r)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    fields = [
        "model",
        "status",
        "hw",
        "n",
        "n_nan",
        "n_pool",
        "match",
        "match_rate",
        "fp32_ref",
        "note",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        mean_m = float(np.mean([r["match_rate"] for r in ok]))
        print(f"DONE ok={len(ok)}/{len(results)} mean_match={mean_m:.4f} -> {OUT_CSV}")
    else:
        print(f"DONE no ok -> {OUT_CSV}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
