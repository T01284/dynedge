"""GPU server probe: MobileNetV3-Small latency + energy (nvidia-smi power integral).

Phase 1.4 — fills GPU-server latency / mJ columns for the probe model.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import models


def sample_power_loop(stop_evt: threading.Event, samples: list[float], interval_s: float = 0.1):
    while not stop_evt.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            samples.append(float(out.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(interval_s)


@torch.no_grad()
def bench_pytorch(model, x, warmup: int, iters: int, device: torch.device):
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    stop = threading.Event()
    powers: list[float] = []
    th = threading.Thread(target=sample_power_loop, args=(stop, powers), daemon=True)
    th.start()
    t0 = time.perf_counter()
    lats = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        a = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - a) * 1e6)  # us
    wall = time.perf_counter() - t0
    stop.set()
    th.join(timeout=2)

    arr = np.asarray(lats, dtype=np.float64)
    mean_w = float(np.mean(powers)) if powers else float("nan")
    # energy ≈ mean_power_W * wall_s / iters → J → mJ
    energy_mj = (mean_w * wall / max(1, iters)) * 1000.0 if powers else float("nan")
    return {
        "backend": "pytorch_eager",
        "latency_us_mean": float(arr.mean()),
        "latency_us_p50": float(np.percentile(arr, 50)),
        "latency_us_p95": float(np.percentile(arr, 95)),
        "latency_us_p99": float(np.percentile(arr, 99)),
        "power_w_mean": mean_w,
        "energy_mj_per_infer": energy_mj,
        "n_power_samples": len(powers),
        "iters": iters,
        "warmup": warmup,
    }


def try_tensorrt_note() -> dict:
    from shutil import which

    trt = which("trtexec")
    if not trt:
        return {"backend": "tensorrt", "status": "skipped", "reason": "trtexec not found"}
    return {"backend": "tensorrt", "status": "available", "trtexec": trt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)

    py = bench_pytorch(model, x, args.warmup, args.iters, device)
    trt = try_tensorrt_note()
    payload = {
        "device": "gpu_server",
        "model": "mobilenet_v3_small",
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "pytorch": py,
        "tensorrt": trt,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "device",
                "backend",
                "model",
                "img_size",
                "batch_size",
                "latency_us_mean",
                "latency_us_p50",
                "latency_us_p95",
                "latency_us_p99",
                "power_w_mean",
                "energy_mj_per_infer",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "device": "gpu_server",
                "backend": py["backend"],
                "model": "mobilenet_v3_small",
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "latency_us_mean": f"{py['latency_us_mean']:.2f}",
                "latency_us_p50": f"{py['latency_us_p50']:.2f}",
                "latency_us_p95": f"{py['latency_us_p95']:.2f}",
                "latency_us_p99": f"{py['latency_us_p99']:.2f}",
                "power_w_mean": f"{py['power_w_mean']:.3f}",
                "energy_mj_per_infer": f"{py['energy_mj_per_infer']:.3f}",
            }
        )
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
