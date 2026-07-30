#!/usr/bin/env python3
"""T527 v4: NCNN INT8 (aciq) accuracy vs NCNN FP32 on 500-image pool.

H1-parallel PTQ metrics (primary = FP32↔INT8 top-1 match).

FP32 models are bake(slimmable→static)+pnnx (see prep_v4_ncnn_baked_pnnx.py).
Runtime forces FP16/BF16 off (A55 asimdhp would otherwise risk overflow).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
EVAL = ROOT / "data" / "calib" / "linux112" / "eval"
CALIB = ROOT / "data" / "calib" / "linux112" / "calib"
OUT_CSV = ROOT / "results" / "t527_int8_acc_500.csv"
OUT_JSON = ROOT / "results" / "t527_int8_acc_500.json"
OUT_LIST = ROOT / "results" / "t527_v4" / "eval500_imagelist.txt"

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FERPLUS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]
EXPW = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
ALIAS = {
    "anger": "anger",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happiness",
    "sad": "sad",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
    "contempt": "contempt",
}


def hw_of(stem: str) -> int:
    if "img64" in stem or "_s64g_" in stem:
        return 64
    if "img96" in stem or "_s96c_" in stem:
        return 96
    return 112


def dataset_of(stem: str) -> str:
    return "ferplus" if "ferplus" in stem else "expw"


def classes_of(stem: str) -> list[str]:
    return FERPLUS if dataset_of(stem) == "ferplus" else EXPW


def folder_label(folder: str) -> str | None:
    name = folder.lower()
    if "_" in name and name.split("_", 1)[0].isdigit():
        name = name.split("_", 1)[1]
    return ALIAS.get(name)


def list_pool(root: Path, source: str) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        parts = p.relative_to(root).parts
        if len(parts) < 2:
            continue
        lab = folder_label(parts[1])
        if lab is None:
            continue
        out.append(
            {
                "path": p,
                "dataset": parts[0],
                "label_name": lab,
                "source": source,
                "rel": str(p.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return out


def build_eval500(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    held = list_pool(EVAL, "eval")
    if len(held) >= n:
        return held[:n]
    calib = list_pool(CALIB, "calib")
    used = {it["rel"] for it in held}
    rest = [it for it in calib if it["rel"] not in used]
    by: dict[str, list[dict]] = {}
    for it in rest:
        by.setdefault(f"{it['dataset']}/{it['label_name']}", []).append(it)
    for k in by:
        random.shuffle(by[k])
    keys = sorted(by)
    topup: list[dict] = []
    i = 0
    need = n - len(held)
    while len(topup) < need and keys:
        progressed = False
        for k in keys:
            if i < len(by[k]):
                topup.append(by[k][i])
                progressed = True
                if len(topup) >= need:
                    break
        if not progressed:
            break
        i += 1
    return held + topup[:need]


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


def load_ncnn(param: Path, bin_path: Path) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.num_threads = 2
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    if net.load_param(str(param)) != 0:
        raise RuntimeError(f"load_param {param}")
    if net.load_model(str(bin_path)) != 0:
        raise RuntimeError(f"load_model {bin_path}")
    return net


def ncnn_logits(net: ncnn.Net, rgb_u8: np.ndarray) -> np.ndarray:
    h, w = rgb_u8.shape[0], rgb_u8.shape[1]
    mat = ncnn.Mat.from_pixels(
        np.ascontiguousarray(rgb_u8), ncnn.Mat.PixelType.PIXEL_RGB, w, h
    )
    mat.substract_mean_normalize([0.0, 0.0, 0.0], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
    ex = net.create_extractor()
    if ex.input("input", mat) != 0:
        raise RuntimeError("ncnn input failed")
    ret, out = ex.extract("output")
    if ret != 0:
        raise RuntimeError(f"ncnn extract {ret}")
    return np.array([out[i] for i in range(out.w * out.h * out.c)], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def eval_one(stem: str, pool: list[dict]) -> dict:
    hw = hw_of(stem)
    gray = "s64g" in stem or "img64" in stem
    ds = dataset_of(stem)
    classes = classes_of(stem)
    fparam = MODEL_DIR / f"{stem}.param"
    fbin = MODEL_DIR / f"{stem}.bin"
    iparam = MODEL_DIR / f"{stem}_int8.param"
    ibin = MODEL_DIR / f"{stem}_int8.bin"
    for p in (fparam, fbin, iparam, ibin):
        if not p.exists():
            return {"model": stem, "status": f"missing:{p.name}"}

    net_fp = load_ncnn(fparam, fbin)
    net_i8 = load_ncnn(iparam, ibin)

    match = 0
    cos_sum = 0.0
    n_ok = 0
    n_lab = 0
    n_nan = 0
    acc_fp = 0
    acc_i8 = 0

    for it in pool:
        rgb = load_rgb_u8(it["path"], hw, gray=gray)
        z_fp = ncnn_logits(net_fp, rgb)
        z_i8 = ncnn_logits(net_i8, rgb)
        if not (np.isfinite(z_fp).all() and np.isfinite(z_i8).all()):
            n_nan += 1
            continue
        p_fp = int(np.argmax(z_fp))
        p_i8 = int(np.argmax(z_i8))
        match += int(p_fp == p_i8)
        cos_sum += cosine(z_fp, z_i8)
        n_ok += 1
        if it["dataset"] == ds and it["label_name"] in classes:
            y = classes.index(it["label_name"])
            acc_fp += int(p_fp == y)
            acc_i8 += int(p_i8 == y)
            n_lab += 1

    if n_ok == 0:
        return {"model": stem, "status": "fail_no_finite", "n_nan": n_nan}

    return {
        "model": stem,
        "status": "ok",
        "dataset": ds,
        "hw": hw,
        "gray": int(gray),
        "method": "aciq",
        "fp32_ref": "ncnn_baked_pnnx",
        "int8_backend": "ncnn",
        "n_eval": n_ok,
        "n_nan": n_nan,
        "n_label": n_lab,
        "n_pool_heldout": sum(1 for it in pool if it["source"] == "eval"),
        "n_pool_calib_topup": sum(1 for it in pool if it["source"] == "calib"),
        "fp32_int8_match": round(match / n_ok, 4),
        "logits_cosine_mean": round(cos_sum / n_ok, 6),
        "fp32_label_top1": round(acc_fp / n_lab, 4) if n_lab else "",
        "int8_label_top1": round(acc_i8 / n_lab, 4) if n_lab else "",
        "note": "fp16_off; bake+pnnx fp32",
    }


def main() -> int:
    global MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="")
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-list", type=Path, default=OUT_LIST)
    args = ap.parse_args()
    MODEL_DIR = args.model_dir
    out_csv = args.out_csv
    out_json = args.out_json
    out_list = args.out_list

    pool = build_eval500(args.n, args.seed)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(it["rel"] for it in pool) + "\n", encoding="utf-8")
    n_h = sum(1 for it in pool if it["source"] == "eval")
    n_c = sum(1 for it in pool if it["source"] == "calib")
    print(f"eval500 n={len(pool)} heldout={n_h} calib_topup={n_c} model_dir={MODEL_DIR}", flush=True)

    stems = sorted(p.stem for p in MODEL_DIR.glob("*.param") if "_int8" not in p.name)
    if args.only:
        stems = [s for s in stems if args.only in s]

    rows = []
    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}", flush=True)
        r = eval_one(stem, pool)
        print(
            f"  status={r.get('status')} match={r.get('fp32_int8_match')} "
            f"cos={r.get('logits_cosine_mean')} "
            f"fp32_top1={r.get('fp32_label_top1')} int8_top1={r.get('int8_label_top1')}",
            flush=True,
        )
        rows.append(r)

    fields = [
        "model",
        "status",
        "dataset",
        "hw",
        "gray",
        "method",
        "fp32_ref",
        "int8_backend",
        "n_eval",
        "n_nan",
        "n_label",
        "n_pool_heldout",
        "n_pool_calib_topup",
        "fp32_int8_match",
        "logits_cosine_mean",
        "fp32_label_top1",
        "int8_label_top1",
        "note",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        mean_m = float(np.mean([r["fp32_int8_match"] for r in ok]))
        print(f"DONE ok={len(ok)}/{len(rows)} mean_match={mean_m:.4f} -> {out_csv}")
    else:
        print(f"DONE no ok -> {out_csv}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
