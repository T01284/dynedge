#!/usr/bin/env python3
"""TensorRT latency bench for Jetson Orin (paper Step 1.1).

Emits READY / RESULT / MEM / BENCH_DONE lines compatible with bench/linux_bench.py.

Example (on Orin):
  python3 bench/trt_bench.py --engine m_fp16.plan --h 64 --w 64 --warmup 50 --iters 1000
  python3 bench/trt_bench.py --engine m_fp16.plan --h 112 --image-dir /path/to/jpg --iters 1000
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

try:
    import tensorrt as trt
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"tensorrt import failed: {e}") from e

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"torch import failed (needed for CUDA buffers): {e}") from e

FER_CLASSES = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)
FW = "orin_trt_1"


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    a = sorted(xs)
    if len(a) == 1:
        return a[0]
    pos = (len(a) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] * (1 - (pos - lo)) + a[hi] * (pos - lo)


def synthetic_nchw(n: int, c: int, h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # [0,1] float32 NCHW — matches common ImageNet-style /255 exports in this repo
    return rng.rand(n, c, h, w).astype(np.float32)


def load_images_nchw(image_dir: Path, h: int, w: int, limit: int) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit(f"PIL required for --image-dir: {e}") from e
    paths = sorted(
        [
            p
            for p in image_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if not paths:
        raise SystemExit(f"no images under {image_dir}")
    paths = paths[: max(1, limit)]
    batch = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC
        batch.append(arr.transpose(2, 0, 1))
    return np.stack(batch, axis=0)


class TrtRunner:
    def __init__(self, engine_path: Path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        blob = engine_path.read_bytes()
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self._bindings: dict[str, torch.Tensor] = {}
        self._input_name = None
        self._output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
        if not self._input_name or not self._output_name:
            raise RuntimeError("engine missing input/output tensor")

    def _ensure(self, shape: tuple[int, ...], dtype=torch.float32) -> None:
        assert self._input_name and self._output_name
        if not self.context.set_input_shape(self._input_name, shape):
            # TRT 10 may return None; also accept when shape already set
            pass
        self.context.set_input_shape(self._input_name, shape)
        in_shape = tuple(self.context.get_tensor_shape(self._input_name))
        out_shape = tuple(self.context.get_tensor_shape(self._output_name))
        if self._input_name not in self._bindings or tuple(self._bindings[self._input_name].shape) != in_shape:
            self._bindings[self._input_name] = torch.empty(in_shape, dtype=dtype, device="cuda")
        if self._output_name not in self._bindings or tuple(self._bindings[self._output_name].shape) != out_shape:
            self._bindings[self._output_name] = torch.empty(out_shape, dtype=dtype, device="cuda")
        self.context.set_tensor_address(self._input_name, int(self._bindings[self._input_name].data_ptr()))
        self.context.set_tensor_address(self._output_name, int(self._bindings[self._output_name].data_ptr()))

    def infer(self, x_nchw: np.ndarray) -> tuple[int, float]:
        """Run one inference; returns (pred, conf)."""
        assert x_nchw.ndim == 4
        self._ensure(tuple(x_nchw.shape))
        inp = self._bindings[self._input_name]
        out = self._bindings[self._output_name]
        inp.copy_(torch.from_numpy(np.ascontiguousarray(x_nchw)).cuda(non_blocking=True))
        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v3 failed")
        self.stream.synchronize()
        logits = out.detach().float().cpu().numpy().reshape(-1)
        pred = int(np.argmax(logits))
        # softmax confidence
        e = np.exp(logits - logits.max())
        conf = float(e[pred] / e.sum())
        return pred, conf


def mem_kb() -> tuple[int, int]:
    # Linux RSS via resource; VMS best-effort from /proc
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is KB
    vms = -1
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmSize:"):
                vms = int(line.split()[1])
                break
    except OSError:
        pass
    return int(rss), int(vms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--device", default="orin")
    ap.add_argument("--backend", default="", help="default: derived from engine name")
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--w", type=int, default=0)
    ap.add_argument("--c", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--image-dir", type=Path, default=None)
    ap.add_argument("--image-limit", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    w = args.w or args.h
    backend = args.backend
    if not backend:
        name = args.engine.name.lower()
        if "int8" in name:
            backend = "trt_int8"
        elif "fp16" in name or "fp32" in name:
            backend = "trt_fp16" if "fp16" in name else "trt_fp32"
        else:
            backend = "trt"

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    runner = TrtRunner(args.engine)
    if args.image_dir:
        imgs = load_images_nchw(args.image_dir, args.h, w, args.image_limit)
    else:
        imgs = synthetic_nchw(max(1, min(8, args.iters)), args.c, args.h, w, args.seed)

    print(
        f"READY device={args.device} fw={FW} backend={backend} "
        f"input={args.h}x{w}x{args.c} classes={len(FER_CLASSES)} trt=1",
        flush=True,
    )

    # warmup
    last_pred, last_conf = 0, 0.0
    for i in range(args.warmup):
        x = imgs[i % len(imgs)][None, ...]
        last_pred, last_conf = runner.infer(x)

    samples: list[float] = []
    for i in range(args.iters):
        x = imgs[i % len(imgs)][None, ...]
        t0 = time.perf_counter()
        last_pred, last_conf = runner.infer(x)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)

    mean = sum(samples) / len(samples)
    p50 = percentile(samples, 0.50)
    p95 = percentile(samples, 0.95)
    p99 = percentile(samples, 0.99)
    cls = FER_CLASSES[last_pred] if 0 <= last_pred < len(FER_CLASSES) else "unk"
    print(
        f"RESULT kind=bench device={args.device} backend={backend} "
        f"warmup={args.warmup} iters={args.iters} input={args.h}x{w}x{args.c} "
        f"latency_us_mean={mean:.1f} latency_us_p50={p50:.1f} "
        f"latency_us_p95={p95:.1f} latency_us_p99={p99:.1f} "
        f"pred={last_pred} class={cls} conf={last_conf:.4f}",
        flush=True,
    )
    rss, vms = mem_kb()
    print(f"MEM rss_kb={rss} vms_kb={vms}", flush=True)
    print("BENCH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    # Allow running as /path/to/board/project without install
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main())
