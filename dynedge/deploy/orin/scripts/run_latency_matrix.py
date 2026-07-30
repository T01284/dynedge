#!/usr/bin/env python3
"""Orin TensorRT latency matrix for 6 FERPlus P5S1 anchors.

Usage:
  python deploy/orin/scripts/run_latency_matrix.py --precision fp16
  python deploy/orin/scripts/run_latency_matrix.py --precision fp16 --limit 1
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[3]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ssh_orin import DEFAULT_HOST, put, run  # noqa: E402
from bench.schema import CSV_FIELDS, ISA_BY_DEVICE, TIER_BY_DEVICE, empty_row  # noqa: E402
from build_engines import ANCHORS, REMOTE, REMOTE_ENG, lock_clocks, push_bench  # noqa: E402

OUT = _ROOT / "results/orin_trt_matrix.csv"
RESULT_RE = re.compile(
    r"RESULT\s+kind=(?P<kind>\w+)\s+device=(?P<device>\S+)\s+backend=(?P<backend>\S+)\s+"
    r"warmup=(?P<warmup>\d+)\s+iters=(?P<iters>\d+)\s+"
    r"(?:input=(?P<input>\S+)\s+)?"
    r"latency_us_mean=(?P<mean>[0-9.]+)\s+latency_us_p50=(?P<p50>[0-9.]+)\s+"
    r"latency_us_p95=(?P<p95>[0-9.]+)\s+latency_us_p99=(?P<p99>[0-9.]+)\s+"
    r"pred=(?P<pred>-?\d+)\s+class=(?P<cls>\w+)\s+conf=(?P<conf>[0-9.]+)"
)
MEM_RE = re.compile(r"MEM\s+rss_kb=(?P<rss_kb>-?\d+)\s+vms_kb=(?P<vms_kb>-?\d+)")


def hw_of(stem: str) -> int:
    if "img64" in stem or "_s64g_" in stem:
        return 64
    if "img96" in stem or "_s96c_" in stem:
        return 96
    return 112


def parse_bench(text: str) -> dict:
    row = empty_row()
    row["device"] = "orin"
    row["tier"] = TIER_BY_DEVICE["orin"]
    row["isa"] = ISA_BY_DEVICE.get("orin", "aarch64")
    for line in text.splitlines():
        m = RESULT_RE.search(line.strip())
        if m:
            d = m.groupdict()
            row.update(
                {
                    "kind": d["kind"],
                    "device": d["device"],
                    "backend": d["backend"],
                    "warmup": int(d["warmup"]),
                    "iters": int(d["iters"]),
                    "input": d.get("input") or "",
                    "latency_us_mean": float(d["mean"]),
                    "latency_us_p50": float(d["p50"]),
                    "latency_us_p95": float(d["p95"]),
                    "latency_us_p99": float(d["p99"]),
                    "pred": int(d["pred"]),
                    "class": d["cls"],
                    "conf": float(d["conf"]),
                }
            )
        m = MEM_RE.search(line.strip())
        if m:
            row["rss_kb"] = int(m.group("rss_kb"))
            row["vms_kb"] = int(m.group("vms_kb"))
    return row


def bench_one(stem: str, precision: str, host: str, warmup: int, iters: int) -> dict:
    eng = f"{REMOTE_ENG}/{stem}_{precision}.plan"
    code, out, _ = run(f"test -s {eng} && echo OK || echo MISS", host)
    if "MISS" in (out or ""):
        raise RuntimeError(f"missing engine {eng}; run build_engines.py first")
    hw = hw_of(stem)
    backend = f"trt_{precision}"
    cmd = (
        f"python3 {REMOTE}/bench/trt_bench.py --engine {eng} --device orin "
        f"--backend {backend} --h {hw} --w {hw} --warmup {warmup} --iters {iters}"
    )
    code, out, err = run(cmd, host, timeout=max(300, iters // 2 + 120))
    text = (out or "") + (err or "")
    if code != 0:
        raise RuntimeError(f"bench failed rc={code}\n{text[-2000:]}")
    row = parse_bench(text)
    row["model"] = stem
    row["fw"] = "orin_trt_1"
    row["backend"] = backend
    row["tier"] = TIER_BY_DEVICE["orin"]
    row["isa"] = ISA_BY_DEVICE.get("orin", "aarch64")
    if not row.get("latency_us_p50"):
        raise RuntimeError(f"no RESULT parsed\n{text[-2000:]}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-lock", action="store_true")
    args = ap.parse_args()

    precisions = [p.strip() for p in args.precision.split(",") if p.strip()]
    anchors = ANCHORS[: args.limit] if args.limit else ANCHORS
    if not args.skip_lock:
        lock_clocks(args.host)
    push_bench(args.host)

    rows: list[dict] = []
    for stem in anchors:
        for p in precisions:
            print(f"\n== {stem} {p} ==", flush=True)
            row = bench_one(stem, p, args.host, args.warmup, args.iters)
            ms = float(row["latency_us_p50"]) / 1000.0
            print(
                f"OK p50={ms:.3f} ms  p99={float(row['latency_us_p99'])/1000:.3f} ms  "
                f"backend={row['backend']}",
                flush=True,
            )
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
