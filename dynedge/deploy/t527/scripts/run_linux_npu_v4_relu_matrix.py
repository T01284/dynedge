#!/usr/bin/env python3
"""T527: G4 ReLU NBG → vpm_run NPU e2e latency (mirrors Hardswish matrix)."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import run_linux_npu_v4_matrix as base

REPO = Path(__file__).resolve().parents[3]
NBG_DIR = REPO / "deploy" / "t527" / "models" / "v4_npu_relu"
OUT_DIR = REPO / "results" / "t527_v4"
REMOTE_BASE = "/root/fer-dynedge/npu/v4_relu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=200)
    ap.add_argument("--zp", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Point shared helpers at ReLU artifacts
    base.NBG_DIR = NBG_DIR
    base.OUT_DIR = OUT_DIR
    base.REMOTE_BASE = REMOTE_BASE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = OUT_DIR / "npu_matrix_summary_relu.csv"
    status = OUT_DIR / "npu_status_relu.txt"

    nbgs = sorted(NBG_DIR.glob("subnet_ferplus_*_uint8.nb"))
    if args.limit:
        nbgs = nbgs[: args.limit]
    if not nbgs:
        print(f"ERR no nb under {NBG_DIR}")
        return 1

    rows: list[dict] = []
    if summary.exists():
        with summary.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {r["model"] for r in rows}

    base.run(
        "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done",
        base.HOST,
    )

    with status.open("a", encoding="utf-8") as sf:
        sf.write(f"===== NPU V4 RELU MATRIX START {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n")
        sf.flush()
        for nbg in nbgs:
            stem = nbg.stem.replace("_uint8", "")
            if stem in done:
                print(f"SKIP {stem}", flush=True)
                continue
            print(f"== NPU-ReLU {stem} ==", flush=True)
            try:
                row = base.bench_one(nbg, args.loops, args.zp)
            except Exception as ex:
                msg = f"FAIL {stem}: {ex}"
                print(msg, flush=True)
                sf.write(msg + "\n")
                sf.flush()
                continue
            rows.append(row)
            done.add(stem)
            with summary.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"OK {stem} p50={row['latency_us_p50']}us ({row['input']})", flush=True)
            sf.write(f"OK {stem} p50={row['latency_us_p50']}\n")
            sf.flush()
        sf.write(
            f"===== NPU V4 RELU MATRIX END {time.strftime('%Y-%m-%dT%H:%M:%S')} "
            f"rows={len(rows)} =====\n"
        )

    # Compact paper column
    cmp_path = OUT_DIR / "npu_vs_ncnn_latency_relu.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "tag", "input", "npu_us_p50", "npu_ms_p50", "note"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "model": r["model"],
                    "tag": r.get("tag", ""),
                    "input": r["input"],
                    "npu_us_p50": r["latency_us_p50"],
                    "npu_ms_p50": round(float(r["latency_us_p50"]) / 1000.0, 3),
                    "note": "g4_relu_nbg; latency_only (T2b match shelved)",
                }
            )
    print(f"DONE -> {summary} ({len(rows)} rows); {cmp_path}")
    return 0 if len(rows) == len(nbgs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
