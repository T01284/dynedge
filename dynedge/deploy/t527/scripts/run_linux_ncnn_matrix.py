#!/usr/bin/env python3
"""Thread matrix for T527 Linux NCNN (1/2/4/8)."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import paramiko

HOST = "BOARD_IP_REDACTED"
REMOTE = "/root/fer-dynedge"
RESULTS = Path(__file__).resolve().parents[3] / "results"
OUT = RESULTS / "t527_linux_ncnn_matrix.csv"
LOG = RESULTS / "t527_linux_ncnn_matrix.log"


def run(c, cmd, timeout=900):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    text = o.read().decode("utf-8", "replace") + e.read().decode("utf-8", "replace")
    code = o.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(text[-2000:])
    return text


def parse(text):
    m = re.search(r"^RESULT\s+(.*)$", text, re.M)
    if not m:
        raise RuntimeError("no RESULT\n" + text[-1500:])
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="root", password=<REDACTED> timeout=10, allow_agent=False, look_for_keys=False)
    logs = []
    rows = []
    try:
        run(
            c,
            "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done",
        )
        models = [
            ("fp32", f"{REMOTE}/models/mobilenetv3_small.param", f"{REMOTE}/models/mobilenetv3_small.bin"),
            ("int8", f"{REMOTE}/models/mobilenetv3_small_int8.param", f"{REMOTE}/models/mobilenetv3_small_int8.bin"),
        ]
        for tag, param, bin_ in models:
            for th in (1, 2, 4, 8):
                cmd = (
                    f"{REMOTE}/bin/fer_ncnn_bench --device t527 --param {param} --bin {bin_} "
                    f"--threads {th} --warmup 30 --iters 300"
                )
                print(f"=== {tag} @{th}t ===")
                text = run(c, cmd)
                logs.append(f"##### {tag} threads={th}\n{text}\n")
                d = parse(text)
                rows.append(
                    {
                        "device": "t527",
                        "os": "buildroot",
                        "model": tag,
                        "threads": th,
                        "latency_us_p50": d.get("latency_us_p50"),
                        "latency_us_mean": d.get("latency_us_mean"),
                        "latency_us_p95": d.get("latency_us_p95"),
                        "backend": d.get("backend"),
                    }
                )
                print(rows[-1])
    finally:
        c.close()
    LOG.write_text("\n".join(logs), encoding="utf-8")
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
