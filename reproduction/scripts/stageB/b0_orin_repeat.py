#!/usr/bin/env python3
"""Run-to-run repeatability of Orin TensorRT latency (FP16 vs INT8).

Alternates precisions so thermal or clock drift hits both equally.
Writes results/stageB/orin_trt_repeat.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "orin" / "scripts"))

from build_engines import lock_clocks  # noqa: E402
from run_latency_matrix import bench_one  # noqa: E402
from ssh_orin import DEFAULT_HOST  # noqa: E402

STEMS = [
    "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_img112",
    "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64",
    "subnet_ferplus_w1_d4444_s112c_seed42_v4_predistill_img112",
]
ROUNDS = 5
OUT = ROOT / "results" / "stageB" / "orin_trt_repeat.csv"


def main() -> int:
    lock_clocks(DEFAULT_HOST)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "precision", "round", "latency_us_p50", "latency_us_p99", "latency_us_mean"])
        for rnd in range(ROUNDS):
            for stem in STEMS:
                for prec in ("fp16", "int8"):
                    r = bench_one(stem, prec, DEFAULT_HOST, warmup=50, iters=1000)
                    w.writerow([stem, prec, rnd, r["latency_us_p50"], r["latency_us_p99"], r["latency_us_mean"]])
                    f.flush()
                    print(f"r{rnd} {stem[15:40]} {prec} p50={r['latency_us_p50']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
