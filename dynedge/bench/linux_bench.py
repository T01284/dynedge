#!/usr/bin/env python3
"""Collect Linux-edge bench lines (T527 / H618 / Orin-compatible schema).

Parses READY/RESULT/MEM/BENCH_DONE from a local command or SSH remote.

Examples:
  python bench/linux_bench.py --cmd "deploy/ncnn_edge/build-stub/fer_ncnn_bench --device stub --warmup 5 --iters 20" --out results/stub.csv
  python bench/linux_bench.py --ssh root@BOARD_IP_REDACTED --remote-cmd "./run_ncnn_probe.sh" --device h618 --out results/h618_ncnn.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.schema import CSV_FIELDS, ISA_BY_DEVICE, TIER_BY_DEVICE

RESULT_RE = re.compile(
    r"RESULT\s+kind=(?P<kind>\w+)\s+device=(?P<device>\S+)\s+backend=(?P<backend>\S+)\s+"
    r"warmup=(?P<warmup>\d+)\s+iters=(?P<iters>\d+)\s+"
    r"(?:input=\S+\s+)?"
    r"latency_us_mean=(?P<mean>[0-9.]+)\s+latency_us_p50=(?P<p50>[0-9.]+)\s+"
    r"latency_us_p95=(?P<p95>[0-9.]+)\s+latency_us_p99=(?P<p99>[0-9.]+)\s+"
    r"pred=(?P<pred>-?\d+)\s+class=(?P<cls>\w+)\s+conf=(?P<conf>[0-9.]+)"
)
MEM_RE = re.compile(r"MEM\s+rss_kb=(?P<rss_kb>-?\d+)\s+vms_kb=(?P<vms_kb>-?\d+)")
READY_RE = re.compile(
    r"READY\s+device=(?P<device>\S+)\s+fw=(?P<fw>\S+)\s+backend=(?P<backend>\S+)\s+"
    r"input=(?P<input>\S+)"
)


def run_local(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        print(out)
        raise SystemExit(f"command failed rc={p.returncode}")
    return out


def run_ssh(host: str, remote_cmd: str) -> str:
    p = subprocess.run(
        ["ssh", host, remote_cmd],
        check=False,
        capture_output=True,
        text=True,
    )
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        print(out)
        raise SystemExit(f"ssh failed rc={p.returncode}")
    return out


def parse_output(text: str) -> dict:
    row: dict = {}
    for line in text.splitlines():
        line = line.strip()
        print(line)
        m = READY_RE.search(line)
        if m:
            row.update(m.groupdict())
        m = RESULT_RE.search(line)
        if m:
            d = m.groupdict()
            row["kind"] = d["kind"]
            row["device"] = d["device"]
            row["backend"] = d["backend"]
            row["warmup"] = int(d["warmup"])
            row["iters"] = int(d["iters"])
            row["latency_us_mean"] = float(d["mean"])
            row["latency_us_p50"] = float(d["p50"])
            row["latency_us_p95"] = float(d["p95"])
            row["latency_us_p99"] = float(d["p99"])
            row["pred"] = int(d["pred"])
            row["class"] = d["cls"]
            row["conf"] = float(d["conf"])
        m = MEM_RE.search(line)
        if m:
            row["rss_kb"] = int(m.group("rss_kb"))
            row["vms_kb"] = int(m.group("vms_kb"))
    return row


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Linux edge bench CSV collector")
    ap.add_argument("--cmd", default="", help="Local shell command to run")
    ap.add_argument("--ssh", default="", help="SSH target user@host")
    ap.add_argument("--remote-cmd", default="", help="Command on remote host")
    ap.add_argument("--parse-file", type=Path, default=None, help="Parse existing log")
    ap.add_argument("--device", default="", help="Override device column")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.parse_file:
        text = args.parse_file.read_text(encoding="utf-8", errors="replace")
    elif args.ssh:
        if not args.remote_cmd:
            raise SystemExit("--remote-cmd required with --ssh")
        text = run_ssh(args.ssh, args.remote_cmd)
    elif args.cmd:
        text = run_local(args.cmd)
    else:
        raise SystemExit("pass --cmd, --ssh/--remote-cmd, or --parse-file")

    row = parse_output(text)
    if not row.get("latency_us_p50"):
        print("ERR no RESULT line parsed", file=sys.stderr)
        return 1
    if args.device:
        row["device"] = args.device
    dev = str(row.get("device", ""))
    row.setdefault("tier", TIER_BY_DEVICE.get(dev, ""))
    row.setdefault("isa", ISA_BY_DEVICE.get(dev, ""))
    row.setdefault("model", row.get("backend", ""))

    print("--- parsed ---")
    for k in CSV_FIELDS:
        if k in row:
            print(f"  {k}: {row[k]}")

    if args.out:
        append_csv(args.out, row)
        print(f"appended {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
