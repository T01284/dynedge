#!/usr/bin/env python3
"""B10: build and bench TensorRT engines on the Orin itself (no SSH in the loop).

For each ONNX in the given order: trtexec FP16 and INT8 (uncalibrated, as in B0), then
bench/trt_bench.py with 50 warmup + 1000 iterations. Rows are appended to the CSV as they
finish, so a relaunch resumes. Benching waits while any training or search job is running.
With --repeat N, N random architectures are re-benched at the end (round=1) to measure drift.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
BUSY = r"train_subnet_distill|train_controller|search_subnets|b10_search|b0_expw_consistency"
RESULT_RE = re.compile(r"latency_us_mean=(?P<mean>[0-9.]+)\s+latency_us_p50=(?P<p50>[0-9.]+)\s+"
                       r"latency_us_p95=(?P<p95>[0-9.]+)\s+latency_us_p99=(?P<p99>[0-9.]+)")
FIELDS = ["arch", "precision", "round", "latency_us_p50", "latency_us_p95", "latency_us_p99",
          "latency_us_mean", "build_s", "iters", "warmup", "ts"]


def sh(cmd: str, timeout: int) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def busy() -> bool:
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if re.search(BUSY, cmd) and "python" in cmd:
            return True
    return False


def wait_idle() -> None:
    while busy():
        print(time.strftime("%F %T"), "waiting: training/search running", flush=True)
        time.sleep(120)


def hw_of(arch: str) -> int:
    return int(re.search(r"_s(\d+)[gc]", arch).group(1))


def build(onnx: Path, eng: Path, prec: str) -> float:
    if eng.is_file() and eng.stat().st_size > 0:
        return 0.0
    flags = "--fp16" if prec == "fp16" else "--int8 --fp16"
    t0 = time.time()
    rc, out = sh(f"{TRTEXEC} --onnx={onnx} --saveEngine={eng} --memPoolSize=workspace:2048M {flags}", 1800)
    if rc != 0:
        raise RuntimeError(f"trtexec rc={rc}\n{out[-1500:]}")
    return time.time() - t0


def bench(eng: Path, prec: str, hw: int, warmup: int, iters: int) -> dict:
    rc, out = sh(f"python3 {ROOT}/bench/trt_bench.py --engine {eng} --device orin --backend trt_{prec} "
                 f"--h {hw} --w {hw} --warmup {warmup} --iters {iters}", 900)
    m = RESULT_RE.search(out)
    if rc != 0 or not m:
        raise RuntimeError(f"bench rc={rc}\n{out[-1500:]}")
    return {k: float(v) for k, v in m.groupdict().items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--order", type=Path, required=True, help="json with an 'order' list of arch keys")
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    order = json.loads(args.order.read_text())["order"]
    args.engine_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.is_file():
        with args.out.open() as f:
            done = {(r["arch"], r["precision"], r["round"]) for r in csv.DictReader(f)}
    wait_idle()
    print(sh("sudo -n nvpmodel -m 2; sudo -n jetson_clocks; sudo -n nvpmodel -q", 60)[1], flush=True)

    jobs = [(a, p, "0") for a in order for p in ("fp16", "int8")]
    if args.repeat:
        rep = random.Random(7).sample(order, args.repeat)
        jobs += [(a, p, "1") for a in rep for p in ("fp16", "int8")]
    fails = 0
    for i, (arch, prec, rnd) in enumerate(jobs):
        if (arch, prec, rnd) in done:
            continue
        wait_idle()
        eng = args.engine_dir / f"{arch}_{prec}.plan"
        try:
            bs = build(args.onnx_dir / f"{arch}.onnx", eng, prec)
            r = bench(eng, prec, hw_of(arch), args.warmup, args.iters)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {arch} {prec}: {e}", flush=True)
            continue
        row = {"arch": arch, "precision": prec, "round": rnd, "latency_us_p50": r["p50"],
               "latency_us_p95": r["p95"], "latency_us_p99": r["p99"], "latency_us_mean": r["mean"],
               "build_s": round(bs, 1), "iters": args.iters, "warmup": args.warmup, "ts": time.strftime("%F %T")}
        new = not args.out.is_file()
        with args.out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{i + 1}/{len(jobs)} {arch} {prec} r{rnd} p50={r['p50']:.1f}us build={bs:.0f}s", flush=True)
    print(f"DONE fails={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
