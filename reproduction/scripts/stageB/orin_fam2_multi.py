#!/usr/bin/env python3
"""B16 on the Orin: re-bench every fam2 TensorRT engine in K separate processes.

On the Orin each trt_bench process lands in one of two latency levels (about 200 us apart
for these sub-ms engines), so a single run per engine mixes the two. This runs K processes
per (arch, precision), interleaved over a shuffled order, and appends one row per run to
the CSV (resumable). Engines come from orin_fam2.sh (models/fam2_trt).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orin_trt_sweep import bench, hw_of, sh, wait_idle  # noqa: E402

FIELDS = ["arch", "precision", "rep", "latency_us_p50", "latency_us_p95", "latency_us_p99",
          "latency_us_mean", "ts"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=Path, required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    order = json.loads(args.order.read_text())["order"]
    done = set()
    if args.out.is_file():
        with args.out.open() as f:
            done = {(r["arch"], r["precision"], r["rep"]) for r in csv.DictReader(f)}
    wait_idle()
    print(sh("sudo -n nvpmodel -m 2; sudo -n jetson_clocks; sudo -n nvpmodel -q", 60)[1], flush=True)

    jobs = [(a, p, str(k)) for k in range(args.k) for a in order for p in ("fp16", "int8")]
    for k in range(args.k):
        block = jobs[k * 2 * len(order):(k + 1) * 2 * len(order)]
        random.Random(100 + k).shuffle(block)
        jobs[k * 2 * len(order):(k + 1) * 2 * len(order)] = block
    fails = 0
    for i, (arch, prec, rep) in enumerate(jobs):
        if (arch, prec, rep) in done:
            continue
        try:
            r = bench(args.engine_dir / f"{arch}_{prec}.plan", prec, hw_of(arch), 50, 1000)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {arch} {prec} rep{rep}: {e}", flush=True)
            continue
        row = {"arch": arch, "precision": prec, "rep": rep, "latency_us_p50": r["p50"],
               "latency_us_p95": r["p95"], "latency_us_p99": r["p99"], "latency_us_mean": r["mean"],
               "ts": time.strftime("%F %T")}
        new = not args.out.is_file()
        with args.out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{i + 1}/{len(jobs)} {arch} {prec} rep{rep} p50={r['p50']:.1f}us", flush=True)
    print(f"DONE fails={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
