#!/usr/bin/env python3
"""Align VIPLite NPU (ACUITY uint8 NBG) vs NCNN FP32/INT8 on linux112 eval.

Uploads all quantized inputs once, runs vpm_run in a board-side loop, then
compares logits with host pyncnn.
"""
from __future__ import annotations

import argparse
import csv
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import ncnn

from ssh_t527 import put, run

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "deploy" / "ncnn_edge" / "models"
EVAL = ROOT / "data" / "calib" / "linux112" / "eval"
NBG = ROOT / "deploy" / "t527" / "models" / "mobilenetv3_small_fer_uint8_real.nb"
META = ROOT / "deploy" / "t527" / "models" / "mobilenetv3_small_fer_uint8_real_nbg_meta.json"
REMOTE = "/root/fer-dynedge/npu/mnv3_fer_eval"
OUT_JSON = ROOT / "results" / "t527_npu_vs_ncnn_acc.json"
OUT_CSV = ROOT / "results" / "t527_npu_vs_ncnn_acc_detail.csv"

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLS_ALIAS = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
    "contempt": "disgust",
}

IMAGENET_MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)
IMAGENET_SCALE_BGR = np.array([0.01743, 0.01751, 0.01712], dtype=np.float32)


def normalize_cls(folder: str) -> str | None:
    name = folder.lower()
    if "_" in name and name.split("_", 1)[0].isdigit():
        name = name.split("_", 1)[1]
    return CLS_ALIAS.get(name)


def list_images(root: Path) -> list[tuple[Path, int]]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        mapped = normalize_cls(p.parent.name)
        if mapped is None:
            continue
        out.append((p, CLASSES.index(mapped)))
    return out


def load_rgb_u8(path: Path) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((112, 112), Image.BILINEAR), dtype=np.uint8
    )


def float_imagenet_nchw(rgb: np.ndarray) -> np.ndarray:
    bgr = rgb[:, :, ::-1].astype(np.float32)
    x = (bgr - IMAGENET_MEAN_BGR) * IMAGENET_SCALE_BGR
    return np.transpose(x, (2, 0, 1))


def float_div255_nchw(rgb: np.ndarray) -> np.ndarray:
    return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))


def quantize_u8(chw: np.ndarray, scale: float, zp: int) -> bytes:
    q = np.rint(chw / scale + zp).astype(np.int32)
    q = np.clip(q, 0, 255).astype(np.uint8)
    return q.tobytes(order="C")


def load_net(param: Path, bin_path: Path) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.num_threads = 4
    if net.load_param(str(param)) != 0:
        raise RuntimeError(f"load_param {param}")
    if net.load_model(str(bin_path)) != 0:
        raise RuntimeError(f"load_model {bin_path}")
    return net


def run_ncnn(net: ncnn.Net, chw: np.ndarray) -> np.ndarray:
    # chw must be C-contiguous; transpose views cause garbage logits via ncnn.Mat.
    mat = ncnn.Mat(np.ascontiguousarray(chw))
    ex = net.create_extractor()
    ex.input("input", mat)
    ret, out = ex.extract("output")
    if ret != 0:
        raise RuntimeError(f"extract failed {ret}")
    return np.array([out[i] for i in range(out.w * out.h * out.c)], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def parse_output_txt(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.splitlines() if x.strip()]
    return np.asarray(vals, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not NBG.exists():
        raise SystemExit(f"missing {NBG}")
    meta = json.loads(META.read_text(encoding="utf-8"))
    q = meta["Inputs"]["input_142"]["quantize"]
    in_scale = float(q["scale"])
    in_zp = int(q["zero_point"])

    pairs = list_images(EVAL)
    if args.limit:
        pairs = pairs[: args.limit]
    if len(pairs) < 5:
        raise SystemExit(f"few eval images: {len(pairs)}")

    fp32 = load_net(MODELS / "mobilenetv3_small.param", MODELS / "mobilenetv3_small.bin")
    int8 = None
    if (MODELS / "mobilenetv3_small_int8_real.param").exists():
        int8 = load_net(
            MODELS / "mobilenetv3_small_int8_real.param",
            MODELS / "mobilenetv3_small_int8_real.bin",
        )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        inputs = td_path / "inputs"
        inputs.mkdir()
        manifest = []
        print(f"preparing {len(pairs)} inputs...", flush=True)
        for i, (path, y) in enumerate(pairs):
            rgb = load_rgb_u8(path)
            x_im = float_imagenet_nchw(rgb)
            x_255 = float_div255_nchw(rgb)
            name = f"{i:04d}"
            (inputs / f"{name}.dat").write_bytes(quantize_u8(x_im, in_scale, in_zp))
            manifest.append(
                {
                    "name": name,
                    "path": str(path.relative_to(EVAL)),
                    "y": y,
                    "x_im": x_im,
                    "x_255": x_255,
                }
            )

        tar_path = td_path / "inputs.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(inputs, arcname="inputs")

        sample = (
            "vpm_run is a framework for testing VIPLite driver. \n"
            "\n"
            "#You may put comment by starting with a '#'.\n"
            "#####################################################A separator makes the list clear.\n"
            "[network]\n"
            f"{REMOTE}/network_binary.nb\n"
            "[input]\n"
            f"{REMOTE}/input.dat\n"
            "\n"
        )
        sample_path = td_path / "sample.txt"
        sample_path.write_text(sample, encoding="utf-8", newline="\n")

        run(f"mkdir -p {REMOTE}", timeout=30)
        put(str(NBG), f"{REMOTE}/network_binary.nb")
        put(str(sample_path), f"{REMOTE}/sample.txt")
        put(str(tar_path), f"{REMOTE}/inputs.tar")

        remote_sh = f"""#!/bin/sh
set -e
cd {REMOTE}
rm -rf inputs outs
mkdir -p outs
tar -xf inputs.tar
chmod 666 /dev/vipcore 2>/dev/null || true
n=0
ok=0
for f in inputs/*.dat; do
  base=$(basename "$f" .dat)
  cp -f "$f" input.dat
  rm -f output_0.txt
  vpm_run -s sample.txt -l 1 -b 0 --save_txt 1 >"outs/${{base}}.log" 2>&1 || true
  if [ -f output_0.txt ]; then
    cp -f output_0.txt "outs/${{base}}.txt"
    ok=$((ok+1))
  fi
  n=$((n+1))
done
echo DONE n=$n ok=$ok
"""
        sh_path = td_path / "run_eval.sh"
        sh_path.write_text(remote_sh.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
        put(str(sh_path), f"{REMOTE}/run_eval.sh")
        print("board NPU eval...", flush=True)
        code, out, err = run(f"sh {REMOTE}/run_eval.sh", timeout=1800)
        print(out + err)
        if "DONE" not in out:
            raise SystemExit(f"board eval failed code={code}")

        # pull outs as tar
        run(f"cd {REMOTE} && tar -cf outs.tar outs", timeout=60)
        local_outs_tar = td_path / "outs.tar"
        # get via ssh cat is heavy; use scp get
        from ssh_t527 import get

        get(f"{REMOTE}/outs.tar", str(local_outs_tar))
        with tarfile.open(local_outs_tar, "r") as tar:
            tar.extractall(td_path)
        outs = td_path / "outs"

        rows = []
        match_npu_fp_im = match_npu_fp_255 = match_i8_fp_255 = 0
        cos_npu_fp_im = []
        cos_npu_fp_255 = []
        cos_i8_fp_255 = []
        acc_fp_im = acc_npu = acc_fp_255 = acc_i8 = 0
        fail = 0

        for m in manifest:
            name = m["name"]
            y = m["y"]
            ot = outs / f"{name}.txt"
            if not ot.exists():
                fail += 1
                continue
            z_npu = parse_output_txt(ot.read_text(encoding="utf-8", errors="replace"))
            if z_npu.size != 7:
                fail += 1
                continue
            z_fp_im = run_ncnn(fp32, m["x_im"])
            z_fp_255 = run_ncnn(fp32, m["x_255"])
            p_npu = int(np.argmax(z_npu))
            p_fp_im = int(np.argmax(z_fp_im))
            p_fp_255 = int(np.argmax(z_fp_255))
            acc_npu += int(p_npu == y)
            acc_fp_im += int(p_fp_im == y)
            acc_fp_255 += int(p_fp_255 == y)
            match_npu_fp_im += int(p_npu == p_fp_im)
            match_npu_fp_255 += int(p_npu == p_fp_255)
            cos_npu_fp_im.append(cosine(z_npu, z_fp_im))
            cos_npu_fp_255.append(cosine(z_npu, z_fp_255))
            row = {
                "path": m["path"],
                "label": CLASSES[y],
                "npu_pred": CLASSES[p_npu],
                "fp32_imagenet_pred": CLASSES[p_fp_im],
                "fp32_div255_pred": CLASSES[p_fp_255],
                "npu_fp32_im_match": int(p_npu == p_fp_im),
                "cos_npu_fp32_im": cos_npu_fp_im[-1],
            }
            if int8 is not None:
                z_i8 = run_ncnn(int8, m["x_255"])
                p_i8 = int(np.argmax(z_i8))
                acc_i8 += int(p_i8 == y)
                match_i8_fp_255 += int(p_i8 == p_fp_255)
                cos_i8_fp_255.append(cosine(z_i8, z_fp_255))
                row["int8_real_pred"] = CLASSES[p_i8]
                row["int8_fp32_255_match"] = int(p_i8 == p_fp_255)
            rows.append(row)

    n = len(rows)
    if n < 5:
        raise SystemExit(f"too few successful NPU outs: {n} fail={fail}")

    summary = {
        "n_eval": n,
        "n_fail": fail,
        "npu_nbg": str(NBG.relative_to(ROOT)),
        "npu_input_quant": {"scale": in_scale, "zero_point": in_zp},
        "note": (
            "Primary: NPU↔NCNN-FP32 top1 match under shared ImageNet-BGR preprocess "
            "(ACUITY). Probe head untrained; label top1 diagnostic. "
            "div255 columns mirror H618 NCNN INT8 protocol."
        ),
        "npu_fp32_imagenet_match": match_npu_fp_im / n,
        "npu_fp32_imagenet_logits_cosine_mean": float(np.mean(cos_npu_fp_im)),
        "npu_fp32_div255_match": match_npu_fp_255 / n,
        "npu_fp32_div255_logits_cosine_mean": float(np.mean(cos_npu_fp_255)),
        "npu_label_top1": acc_npu / n,
        "fp32_imagenet_label_top1": acc_fp_im / n,
        "fp32_div255_label_top1": acc_fp_255 / n,
    }
    if int8 is not None and cos_i8_fp_255:
        summary.update(
            {
                "int8_real_fp32_div255_match": match_i8_fp_255 / n,
                "int8_real_fp32_div255_logits_cosine_mean": float(np.mean(cos_i8_fp_255)),
                "int8_real_label_top1": acc_i8 / n,
            }
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
