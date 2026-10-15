#!/usr/bin/env python3
"""B0: TensorRT FP16/INT8 latency for every real v4 subnet ONNX.

The six anchors already on the board stay; this adds the other architectures
that exist as non-empty ONNX (formal LUT exports and the two extra predistill
nets). Empty git-lfs placeholders are skipped.

Writes results/stageB/orin_trt_extended.csv incrementally.
"""
from __future__ import annotations

import csv
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "orin" / "scripts"))

from build_engines import build_one, ensure_remote_tree, lock_clocks, push_bench  # noqa: E402
from run_latency_matrix import bench_one  # noqa: E402
from ssh_orin import DEFAULT_HOST  # noqa: E402

ARCH_RE = re.compile(r"(w[0-9.]+_d\d{4}_s\d+[gc])")
OUT = ROOT / "results" / "stageB" / "orin_trt_extended.csv"
PREC = ("fp16", "int8")

# Earlier dirs win, and a ferplus file beats an expw file of the same arch.
SOURCES = (
    ROOT / "results" / "h618_onnx_v4",
    ROOT / "results" / "h618_onnx_v4_formal",
    ROOT / "results" / "h618_onnx_v4_formal_expw",
    ROOT / "results" / "h618_onnx_v4_flops_ctrl",
)


def select_onnx() -> list[tuple[str, Path]]:
    picked: dict[str, Path] = {}
    for folder in SOURCES:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("subnet_*.onnx")):
            if path.stat().st_size < 10_000:
                continue
            m = ARCH_RE.search(path.name)
            if not m:
                continue
            key = m.group(1)
            prev = picked.get(key)
            if prev is None or ("ferplus" in path.name and "ferplus" not in prev.name):
                picked[key] = path
    return sorted(picked.items())


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {(r["arch"], r["precision"]) for r in csv.DictReader(f) if r.get("latency_us_p50")}


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    host = DEFAULT_HOST
    pairs = select_onnx()
    print(f"architectures with real ONNX: {len(pairs)}", flush=True)
    for arch, path in pairs:
        print(f"  {arch}  {path.relative_to(ROOT)}", flush=True)
    lock_clocks(host)
    ensure_remote_tree(host)
    push_bench(host)
    have = done_keys(OUT)
    fails = 0
    for arch, path in pairs:
        stem = path.stem
        for prec in PREC:
            if (arch, prec) in have:
                print(f"skip {arch} {prec}", flush=True)
                continue
            print(f"\n== {arch} {prec} ==", flush=True)
            try:
                build_one(stem, prec, host, 2048, path)
                row = bench_one(stem, prec, host, warmup=50, iters=1000)
                row = {
                    "arch": arch,
                    "stem": stem,
                    "precision": prec,
                    "onnx": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "latency_us_p50": row.get("latency_us_p50"),
                    "latency_us_p95": row.get("latency_us_p95"),
                    "latency_us_p99": row.get("latency_us_p99"),
                    "latency_us_mean": row.get("latency_us_mean"),
                    "iters": row.get("iters"),
                    "warmup": row.get("warmup"),
                }
                append_row(OUT, row)
                print(
                    f"  p50={row['latency_us_p50']} us  p99={row['latency_us_p99']} us",
                    flush=True,
                )
            except Exception:
                fails += 1
                traceback.print_exc()
    print(f"\nDONE fails={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
