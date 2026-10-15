#!/usr/bin/env python3
"""B13 MCU leg (offline): ESP-DL G3 anchors on the full test set, without a board.

The deployed chain (tools/espdl_ptq_d2222.py, firmware ImagePreprocessor) feeds
PIL luma -> RGB -> PIL bilinear resize -> per-channel ImageNet normalisation.
Training evaluates gray/64 models on the mean of the three *normalised* channels,
resized with F.interpolate (no antialias). This script measures what that costs:

  FP32 (onnxruntime) and INT8 (ESP-PPQ TorchExecutor, same espdl_setting as the
  deployed export) x eval prep {pil, train} x calibration prep {pil, train},
  per target {esp32p4, esp32s3}, on all 3153 FER+ (or 9094 ExpW) test images.

The board-vs-host logits gate (8/8) still has to be repeated on the boards for any
re-exported model; this script only settles the host-side accuracy.

  python scripts/stageB/b14_mcu_offline_acc.py [--dataset ferplus|expw] [--only w035_d2222]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "deploy/mcu_espdl/models"
CKPT_DIR = ROOT / "results/from_autodl/formal_v4/subnets_v4_relu_mcu"
OUT = ROOT / "results/stageB/mcu_offline_acc"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
HW = 64
DATA = {
    "ferplus": {"test": ROOT / "data/ferplus112/test", "n": 3153, "glob": "*.png",
                "calib": ROOT / "data/calib/ferplus_mcu64/train",
                "keys": ("w035_d2222", "w035_d2233", "w035_d2334", "w05_d3333")},
    "expw": {"test": ROOT / "data/expw_faithful/aligned/test", "n": 9094, "glob": "*",
             "calib": ROOT / "data/calib/ferplus_mcu64/train",
             "calib_own": ROOT / "data/expw_faithful/aligned/train",
             "keys": ("w035_d2222", "w035_d2233", "w035_d2344", "w05_d3333")},
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
FIELDS = ["dataset", "model", "target", "dtype", "calib", "prep", "n", "acc", "pytorch_acc", "seconds"]


def prep_pil(p: Path) -> np.ndarray:
    img = Image.open(p).convert("L").convert("RGB").resize((HW, HW), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (a - MEAN) / STD


def prep_train(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy((a.transpose(2, 0, 1) - MEAN) / STD)[None]
    x = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    if x.shape[-1] != HW or x.shape[-2] != HW:
        x = F.interpolate(x, size=(HW, HW), mode="bilinear", align_corners=False)
    return x[0].numpy()


PREPS = {"pil": prep_pil, "train": prep_train}


def list_images(root: Path, glob: str) -> list[tuple[Path, int]]:
    items = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        label = int(d.name.split("_", 1)[0])
        items += [(f, label) for f in sorted(d.glob(glob)) if f.suffix.lower() in IMG_EXT]
    return items


def load_set(items, prep: str) -> np.ndarray:
    fn = PREPS[prep]
    return np.stack([fn(p) for p, _ in items]).astype(np.float32)


def calib_set(root: Path, prep: str, limit: int = 256) -> torch.Tensor:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)[:limit]
    return torch.from_numpy(np.stack([PREPS[prep](p) for p in paths]).astype(np.float32))


def acc_of(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(1) == labels).mean())


def run_fp32(onnx: Path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return np.concatenate([s.run(None, {name: X[i:i + 1]})[0] for i in range(len(X))])


def quantize(onnx: Path, target: str, calib: torch.Tensor, tmp: Path):
    from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx
    loader = torch.utils.data.DataLoader(calib, batch_size=8, shuffle=False)
    return espdl_quantize_onnx(
        onnx_import_file=str(onnx), espdl_export_file=str(tmp / f"{onnx.stem}_{target}.espdl"),
        calib_dataloader=loader, calib_steps=32, input_shape=[1, 3, HW, HW], target=target,
        num_of_bits=8, collate_fn=lambda b: b.to("cpu"), setting=QuantizationSettingFactory.espdl_setting(),
        device="cpu", error_report=False, skip_export=False, export_test_values=False, verbose=0)


def run_int8(graph, X: np.ndarray) -> np.ndarray:
    from esp_ppq.executor.torch import TorchExecutor
    ex = TorchExecutor(graph=graph, device="cpu")
    out = []
    for i in range(len(X)):
        y = ex.forward(inputs=[torch.from_numpy(X[i:i + 1])])[0]
        out.append(np.asarray(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y).reshape(1, -1))
    return np.concatenate(out)


def check_host_ref(graph, onnx: Path, target: str) -> float:
    """Max |INT8 logit| difference against the deployed host_ref (same PIL prep, same calib)."""
    ref = MODELS_DIR / f"{onnx.stem}_{target}_host_ref.json"
    if not ref.exists():
        return float("nan")
    rows = json.loads(ref.read_text(encoding="utf-8"))["samples"][:8]
    paths = [Path(r["path"]) for r in rows]
    paths = [ROOT / Path(*p.parts[p.parts.index("data"):]) for p in paths]
    X = np.stack([prep_pil(p) for p in paths]).astype(np.float32)
    got = run_int8(graph, X)
    return float(np.max(np.abs(got - np.array([r["int8_logits"] for r in rows], dtype=np.float32))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=tuple(DATA), default="ferplus")
    ap.add_argument("--only", default="")
    ap.add_argument("--targets", default="esp32p4,esp32s3")
    args = ap.parse_args()
    cfg = DATA[args.dataset]
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"summary_{args.dataset}.csv"
    done = set()
    if out_csv.exists():
        done = {(r["model"], r["target"], r["dtype"], r["calib"], r["prep"])
                for r in csv.DictReader(out_csv.open(encoding="utf-8"))}
    new = not out_csv.exists()
    f = out_csv.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    items = list_images(cfg["test"], cfg["glob"])
    if len(items) != cfg["n"]:
        raise SystemExit(f"{cfg['test']}: {len(items)} images, expected {cfg['n']}")
    labels = np.array([lab for _, lab in items])
    X = {p: load_set(items, p) for p in PREPS}
    print(f"loaded {len(items)} test images x {len(X)} preps", flush=True)
    calibs = {("deployed" if k == "calib" else "own") + f"_{p}": (cfg[k], p)
              for k in ("calib", "calib_own") if k in cfg for p in PREPS}

    def emit(row):
        w.writerow(row); f.flush()
        print(f"{row['model']} {row['target']} {row['dtype']} calib={row['calib']} prep={row['prep']} "
              f"acc={row['acc']:.4f} (pytorch {row['pytorch_acc']:.4f})", flush=True)

    for key in cfg["keys"]:
        if args.only and args.only not in key:
            continue
        onnx = MODELS_DIR / f"g3_{key}_s64g_{args.dataset}.onnx"
        wk, dk = key.replace("w035", "w0.35").replace("w05", "w0.5").split("_")
        pt = json.loads((CKPT_DIR / f"subnet_{args.dataset}_{wk}_{dk}_s64g_seed42_v4_relu_mcu.json")
                        .read_text(encoding="utf-8"))["test_acc"]
        for prep in PREPS:
            if (onnx.stem, "-", "fp32", "-", prep) in done:
                continue
            t0 = time.time()
            emit({"dataset": args.dataset, "model": onnx.stem, "target": "-", "dtype": "fp32", "calib": "-",
                  "prep": prep, "n": len(labels), "acc": acc_of(run_fp32(onnx, X[prep]), labels),
                  "pytorch_acc": pt, "seconds": round(time.time() - t0, 1)})
        for target in args.targets.split(","):
            for cname, (croot, cprep) in calibs.items():
                todo = [p for p in PREPS if (onnx.stem, target, "int8", cname, p) not in done]
                if not todo:
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    graph = quantize(onnx, target, calib_set(croot, cprep), Path(tmp))
                    if cname == "deployed_pil":
                        d = check_host_ref(graph, onnx, target)
                        print(f"  reproduce deployed host_ref: max|dlogit|={d:.4g}", flush=True)
                    for prep in todo:
                        t0 = time.time()
                        emit({"dataset": args.dataset, "model": onnx.stem, "target": target, "dtype": "int8",
                              "calib": cname, "prep": prep, "n": len(labels),
                              "acc": acc_of(run_int8(graph, X[prep]), labels), "pytorch_acc": pt,
                              "seconds": round(time.time() - t0, 1)})
    f.close()
    print(f"-> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
