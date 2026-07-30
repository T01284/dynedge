#!/usr/bin/env python3
"""Deploy fer_ncnn_bench + models to T527 and run FP32/INT8 formal probes."""
from __future__ import annotations

import csv
import re
import statistics
import time
from pathlib import Path

import paramiko
from scp import SCPClient

HOST = "BOARD_IP_REDACTED"
USER = "root"
PASSWORD = <REDACTED>
REPO = Path(__file__).resolve().parents[3]
BIN = REPO / "deploy" / "ncnn_edge" / "build-linux-aarch64" / "fer_ncnn_bench"
MODELS = REPO / "deploy" / "ncnn_edge" / "models"
REMOTE_DIR = "/root/fer-dynedge"
RESULTS = REPO / "results"


def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=<REDACTED> timeout=15, allow_agent=False, look_for_keys=False)
    return c


def run(c, cmd: str, timeout: int = 900) -> str:
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    text = out + (("\n" + err) if err.strip() else "")
    if code != 0:
        raise RuntimeError(f"exit={code}\ncmd={cmd}\n{text[-4000:]}")
    return text


def parse_result(text: str) -> dict:
    # RESULT device=... backend=... latency_us_p50=...
    m = re.search(r"^RESULT\s+(.*)$", text, re.M)
    if not m:
        # fallback: collect latencies if any
        raise RuntimeError("no RESULT line\n" + text[-3000:])
    fields = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            fields[k] = v
    return fields


def write_csv(path: Path, fields: dict, extra: dict) -> None:
    row = {**fields, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow(row)


def main() -> None:
    if not BIN.is_file():
        raise SystemExit(f"missing {BIN}")
    c = ssh()
    try:
        print("=== deploy ===")
        run(c, f"mkdir -p {REMOTE_DIR}/models {REMOTE_DIR}/bin")
        with SCPClient(c.get_transport()) as scp:
            scp.put(str(BIN), f"{REMOTE_DIR}/bin/fer_ncnn_bench")
            for name in [
                "mobilenetv3_small.param",
                "mobilenetv3_small.bin",
                "mobilenetv3_small_int8.param",
                "mobilenetv3_small_int8.bin",
                "mobilenetv3_small_int8_real.param",
                "mobilenetv3_small_int8_real.bin",
            ]:
                scp.put(str(MODELS / name), f"{REMOTE_DIR}/models/{name}")
        run(c, f"chmod +x {REMOTE_DIR}/bin/fer_ncnn_bench")
        print(run(c, f"file {REMOTE_DIR}/bin/fer_ncnn_bench; {REMOTE_DIR}/bin/fer_ncnn_bench --help 2>&1 | head -n 20"))

        # lock perf
        lock = r"""
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f 2>/dev/null || true; done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
"""
        print("=== lock ===")
        print(run(c, lock))

        jobs = [
            (
                "fp32",
                f"{REMOTE_DIR}/bin/fer_ncnn_bench --device t527 --param {REMOTE_DIR}/models/mobilenetv3_small.param --bin {REMOTE_DIR}/models/mobilenetv3_small.bin --threads 4 --warmup 50 --iters 1000",
                RESULTS / "t527_linux_fp32_probe.csv",
                RESULTS / "t527_linux_fp32_probe.log",
                {"model": "mobilenetv3_small_fp32", "tier": "linux_cpu"},
            ),
            (
                "int8",
                f"{REMOTE_DIR}/bin/fer_ncnn_bench --device t527 --param {REMOTE_DIR}/models/mobilenetv3_small_int8.param --bin {REMOTE_DIR}/models/mobilenetv3_small_int8.bin --threads 4 --warmup 50 --iters 1000",
                RESULTS / "t527_linux_int8_probe.csv",
                RESULTS / "t527_linux_int8_probe.log",
                {"model": "mobilenetv3_small_int8", "tier": "linux_cpu"},
            ),
            (
                "int8_real",
                f"{REMOTE_DIR}/bin/fer_ncnn_bench --device t527 --param {REMOTE_DIR}/models/mobilenetv3_small_int8_real.param --bin {REMOTE_DIR}/models/mobilenetv3_small_int8_real.bin --threads 4 --warmup 50 --iters 1000",
                RESULTS / "t527_linux_int8_real_probe.csv",
                RESULTS / "t527_linux_int8_real_probe.log",
                {"model": "mobilenetv3_small_int8_real", "tier": "linux_cpu"},
            ),
        ]

        for tag, cmd, csv_path, log_path, extra in jobs:
            print(f"=== run {tag} ===")
            t0 = time.time()
            text = run(c, cmd, timeout=1200)
            dt = time.time() - t0
            log_path.write_text(text, encoding="utf-8")
            fields = parse_result(text)
            write_csv(csv_path, fields, {**extra, "wall_s": f"{dt:.1f}", "os": "buildroot"})
            print(f"{tag}: {fields} -> {csv_path.name}")
    finally:
        c.close()
    print("DONE")


if __name__ == "__main__":
    main()
