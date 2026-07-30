#!/usr/bin/env python3
"""Save MobileNetV2 vpm_run smoke timings to results/."""
from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

import paramiko

HOST = "BOARD_IP_REDACTED"
OUT = Path(__file__).resolve().parents[3] / "results" / "t527_linux_npu_mobilenetv2_smoke.csv"
RAW = Path(__file__).resolve().parents[3] / "results" / "t527_linux_npu_mobilenetv2_smoke.log"


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="root", password=<REDACTED> timeout=10, allow_agent=False, look_for_keys=False)
    cmd = r"""
chmod 666 /dev/vipcore 2>/dev/null || true
cd /etc/npu/sample_viplite/MobileNet_V2
vpm_run -s sample.txt -l 200 -b 1 2>&1
"""
    _, out, err = c.exec_command(cmd, timeout=300)
    text = out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace")
    c.close()
    RAW.write_text(text, encoding="utf-8")
    times = [int(x) for x in re.findall(r"profile inference time=(\d+)us", text)]
    if not times:
        raise SystemExit("no timings parsed")
    # drop first as warm-up-ish if many
    use = times[10:] if len(times) > 20 else times
    p50 = statistics.median(use)
    mean = statistics.mean(use)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "device",
                "backend",
                "model",
                "iters",
                "latency_us_p50",
                "latency_us_mean",
                "latency_us_min",
                "latency_us_max",
                "note",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "device": "t527",
                "backend": "viplite_npu",
                "model": "MobileNetV2_ImageNet_stock_nb",
                "iters": len(use),
                "latency_us_p50": int(p50),
                "latency_us_mean": int(mean),
                "latency_us_min": min(use),
                "latency_us_max": max(use),
                "note": "stock BSP sample; not FER MNV3; formal FER NBG TBD",
            }
        )
    print(f"OK n={len(use)} p50_us={int(p50)} mean_us={int(mean)} -> {OUT}")


if __name__ == "__main__":
    main()
