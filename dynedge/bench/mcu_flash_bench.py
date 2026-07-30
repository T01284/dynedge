#!/usr/bin/env python3
"""Flash-trigger-collect helper for fer-dynedge MCU firmware.

Aligns CSV fields with Linux bench schema (device, latency_us_p50/p95/p99, ...).

Examples:
  python bench/mcu_flash_bench.py --port SERIAL_PORT --device esp32s3 --bench
  python bench/mcu_flash_bench.py --port SERIAL_PORT --cmd "CMD INFO"
  python bench/mcu_flash_bench.py --port SERIAL_PORT --warmup 50 --iters 200 --out results/mcu.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.schema import CSV_FIELDS, ISA_BY_DEVICE, TIER_BY_DEVICE

try:
    import serial
except ImportError as e:
    raise SystemExit("pip install pyserial") from e

RESULT_RE = re.compile(
    r"RESULT\s+kind=(?P<kind>\w+).*?"
    r"(?:latency_us(?:_mean)?=(?P<mean>\d+)\s+)?"
    r"(?:latency_us_p50=(?P<p50>\d+)\s+)?"
    r"(?:latency_us_p95=(?P<p95>\d+)\s+)?"
    r"(?:latency_us_p99=(?P<p99>\d+)\s+)?"
    r"(?:latency_us=(?P<single>\d+)\s+)?"
    r"pred=(?P<pred>-?\d+)\s+class=(?P<cls>\w+)\s+conf=(?P<conf>[0-9.]+)"
)
MEM_RE = re.compile(
    r"MEM\s+heap_free=(?P<heap_free>\d+).*?psram_free=(?P<psram_free>\d+)\s+psram_ok=(?P<psram_ok>\d+)"
)
READY_RE = re.compile(r"READY\s+device=(?P<device>\S+)")


def open_port(port: str, baud: int = 115200, timeout: float = 1.0):
    ser = serial.Serial(port, baudrate=baud, timeout=timeout)
    time.sleep(0.3)
    ser.reset_input_buffer()
    return ser


def read_until(ser, predicates, timeout_s: float = 60.0) -> list[str]:
    deadline = time.time() + timeout_s
    lines: list[str] = []
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        print(line)
        lines.append(line)
        for pred in predicates:
            if pred(line):
                return lines
    raise TimeoutError(f"timeout after {timeout_s}s; last lines: {lines[-5:]}")


def send_cmd(ser, cmd: str) -> None:
    ser.write((cmd.strip() + "\n").encode("utf-8"))
    ser.flush()


def parse_result(lines: list[str]) -> dict:
    out: dict = {}
    for line in lines:
        m = RESULT_RE.search(line)
        if m:
            d = m.groupdict()
            out["kind"] = d["kind"]
            out["pred"] = int(d["pred"])
            out["class"] = d["cls"]
            out["conf"] = float(d["conf"])
            if d.get("p50"):
                out["latency_us_p50"] = int(d["p50"])
                out["latency_us_p95"] = int(d["p95"])
                out["latency_us_p99"] = int(d["p99"])
                out["latency_us_mean"] = int(d["mean"]) if d.get("mean") else None
            if d.get("single"):
                out["latency_us"] = int(d["single"])
        mm = MEM_RE.search(line)
        if mm:
            out["heap_free"] = int(mm.group("heap_free"))
            out["psram_free"] = int(mm.group("psram_free"))
            out["psram_ok"] = int(mm.group("psram_ok"))
        rm = READY_RE.search(line)
        if rm:
            out["device"] = rm.group("device")
    return out


def append_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="MCU serial bench collector")
    ap.add_argument("--port", required=True, help="Serial port, e.g. SERIAL_PORT or /dev/ttySERIAL")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--device", default="", help="Logical device name for CSV (esp32s3/esp32p4)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--bench", action="store_true", help="Run CMD BENCH")
    ap.add_argument("--cmd", default="", help="Raw command to send instead of bench")
    ap.add_argument("--out", type=Path, default=None, help="Append CSV path")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    ser = open_port(args.port, args.baud)
    try:
        send_cmd(ser, "CMD INFO")
        info_lines = read_until(ser, [lambda l: l.startswith("MEM ")], timeout_s=10)
        info = parse_result(info_lines)

        if args.cmd:
            send_cmd(ser, args.cmd)
            lines = read_until(
                ser,
                [lambda l: l.startswith("RESULT ") or l.startswith("OK ") or l.startswith("ERR ")],
                timeout_s=args.timeout,
            )
        elif args.bench:
            send_cmd(ser, f"CMD BENCH {args.warmup} {args.iters}")
            lines = read_until(ser, [lambda l: l.startswith("BENCH_DONE")], timeout_s=args.timeout)
        else:
            print("Nothing to do; pass --bench or --cmd", file=sys.stderr)
            return 2

        result = parse_result(info_lines + lines)
        if args.device:
            result["device"] = args.device
        result.setdefault("device", info.get("device", "unknown"))
        result["warmup"] = args.warmup
        result["iters"] = args.iters
        result["backend"] = result.get("backend") or "probe_int8"
        # Prefer size from READY line if present
        input_sz = "64x64x3"
        for line in info_lines + lines:
            if "input=" in line and "READY" in line:
                m = re.search(r"input=(\d+)x(\d+)x(\d+)", line)
                if m:
                    input_sz = f"{m.group(1)}x{m.group(2)}x{m.group(3)}"
                    break
            if "model=" in line:
                mm = re.search(r"model=(\S+)", line)
                if mm:
                    result["backend"] = mm.group(1)
        result["input"] = input_sz
        result.setdefault("kind", "bench" if args.bench else "cmd")

        print("--- parsed ---")
        for k in sorted(result):
            print(f"  {k}: {result[k]}")

        if args.out:
            dev = str(result.get("device", ""))
            result.setdefault("tier", TIER_BY_DEVICE.get(dev, "MCU"))
            result.setdefault("isa", ISA_BY_DEVICE.get(dev, ""))
            result.setdefault("model", result.get("backend", ""))
            result.setdefault("fw", "")
            result.setdefault("rss_kb", "")
            result.setdefault("vms_kb", "")
            append_csv(args.out, result, CSV_FIELDS)
            print(f"appended {args.out}")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
