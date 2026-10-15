#!/usr/bin/env python3
"""C4: calibrated TensorRT INT8 for the Table II mid subnet, on the Orin.

Builds an entropy-calibrated INT8 engine of the seed-42 predistill ONNX
(w0.5_d3333@112), scores the full FER+ test set, and times it against the
uncalibrated engine with the same bench (warmup 50, 1000 iterations, ten
processes, keep the minimum P50).

Calibration and accuracy use the training eval preprocessing: resize to 112,
ToTensor, ImageNet normalization. The ONNX itself has no normalization.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt
from torchvision import datasets, transforms

ROOT = Path("<gpu-workdir>/fer-dynedge")
STEM = "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_img112"
ONNX = ROOT / "models/v4_predistill_onnx" / f"{STEM}.onnx"
ENG_DIR = ROOT / "models/v4_predistill_trt"
OUT = ROOT / "results/stageB/c4_orin_int8"
BENCH = ROOT / "bench/trt_bench.py"
DATA = ROOT / "data/ferplus"
CALIB_N = 256
CALIB_SEED = 0
K_LAT = 10


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, images: np.ndarray, cache: Path):
        super().__init__()
        self.images = images
        self.cache = cache
        self.i = 0
        self.device = torch.empty((1, 3, 112, 112), dtype=torch.float32, device="cuda")

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.i >= len(self.images):
            return None
        self.device.copy_(torch.from_numpy(self.images[self.i : self.i + 1]))
        self.i += 1
        return [int(self.device.data_ptr())]

    def read_calibration_cache(self):
        if self.cache.is_file() and self.cache.stat().st_size > 0:
            return self.cache.read_bytes()
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache.write_bytes(cache)


def eval_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def collect(ds, n: int | None, seed: int) -> np.ndarray:
    idx = list(range(len(ds)))
    if n is not None and n < len(idx):
        rng = random.Random(seed)
        idx = rng.sample(idx, n)
    xs = []
    for i in idx:
        x, _ = ds[i]
        xs.append(x.numpy())
    return np.stack(xs, axis=0).astype(np.float32)


def build_calibrated(images: np.ndarray, engine: Path, cache: Path) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(ONNX.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("onnx parse failed\n" + "\n".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = EntropyCalibrator(images, cache)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("calibrated engine build returned None")
    engine.write_bytes(blob)


def trtexec_uncalibrated(engine: Path) -> None:
    if engine.is_file() and engine.stat().st_size > 0:
        return
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={ONNX}",
        f"--saveEngine={engine}",
        "--memPoolSize=workspace:2048M",
        "--int8",
        "--fp16",
    ]
    subprocess.run(cmd, check=True)


def accuracy(engine: Path, images: np.ndarray, labels: np.ndarray) -> dict:
    sys.path.insert(0, str(ROOT))
    from bench.trt_bench import TrtRunner

    runner = TrtRunner(engine)
    correct = 0
    for i in range(len(images)):
        pred, _ = runner.infer(images[i : i + 1])
        correct += int(pred == int(labels[i]))
    return {"n": int(len(images)), "correct": int(correct), "acc": correct / len(images)}


def latency(engine: Path) -> dict:
    p50 = []
    p99 = []
    for _ in range(K_LAT):
        proc = subprocess.run(
            [sys.executable, str(BENCH), "--engine", str(engine), "--h", "112", "--warmup", "50", "--iters", "1000"],
            check=True, capture_output=True, text=True,
        )
        line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
        fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        p50.append(float(fields["latency_us_p50"]))
        p99.append(float(fields["latency_us_p99"]))
    return {
        "k": K_LAT,
        "p50_us": p50,
        "p99_us": p99,
        "p50_min_us": min(p50),
        "p50_median_us": float(np.median(p50)),
        "p99_at_min_p50_us": p99[int(np.argmin(p50))],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ENG_DIR.mkdir(parents=True, exist_ok=True)
    print("tensorrt", trt.__version__, flush=True)
    if not ONNX.is_file():
        raise SystemExit(f"missing onnx {ONNX}")

    tfm = eval_transform()
    train = datasets.ImageFolder(DATA / "train", transform=tfm)
    test = datasets.ImageFolder(DATA / "test", transform=tfm)
    print("classes", test.classes, "test", len(test), "train", len(train), flush=True)
    calib = collect(train, CALIB_N, CALIB_SEED)
    labels = np.array([test.targets[i] for i in range(len(test))], dtype=np.int64)
    # ImageFolder keeps samples in folder order; targets aligns with indices.
    test_x = collect(test, None, 0)

    uncal = ENG_DIR / f"{STEM}_int8.plan"
    cal = ENG_DIR / f"{STEM}_int8_calib.plan"
    cache = OUT / "calib.cache"
    print("uncalibrated engine", uncal, "exists", uncal.is_file(), flush=True)
    trtexec_uncalibrated(uncal)
    if not cal.is_file():
        print("building calibrated engine", flush=True)
        build_calibrated(calib, cal, cache)
    else:
        print("calibrated engine exists", flush=True)

    summary = {
        "onnx": str(ONNX),
        "trt": trt.__version__,
        "calib_n": CALIB_N,
        "calib_seed": CALIB_SEED,
        "test_n": len(test),
        "classes": test.classes,
        "uncalibrated": {"engine": str(uncal), "acc": accuracy(uncal, test_x, labels), "latency": latency(uncal)},
        "calibrated": {"engine": str(cal), "acc": accuracy(cal, test_x, labels), "latency": latency(cal)},
    }
    u = summary["uncalibrated"]["latency"]["p50_min_us"]
    c = summary["calibrated"]["latency"]["p50_min_us"]
    summary["p50_min_rel"] = c / u - 1
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "uncal_acc": summary["uncalibrated"]["acc"],
        "cal_acc": summary["calibrated"]["acc"],
        "uncal_p50_min_ms": u / 1000,
        "cal_p50_min_ms": c / 1000,
        "rel": summary["p50_min_rel"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
