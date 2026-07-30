#!/usr/bin/env python3
"""Deploy FER MobileNetV3 uint8 NBG to T527 and capture vpm_run latency."""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import tempfile
from pathlib import Path

from ssh_t527 import put, run

HOST = "BOARD_IP_REDACTED"
REPO = Path(__file__).resolve().parents[3]
REMOTE_DIR = "/root/fer-dynedge/npu/mnv3_fer"

# nbg_meta: u8 NCHW 1x3x112x112, zp=113 (synth and real share input zp)
IN_ELEMS = 1 * 3 * 112 * 112
ZERO_POINT = 113


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real",
        action="store_true",
        help="use FER/ExpW real-calib NBG (128 imgs)",
    )
    args = ap.parse_args()
    if args.real:
        local_nbg = REPO / "deploy" / "t527" / "models" / "mobilenetv3_small_fer_uint8_real.nb"
        out = REPO / "results" / "t527_linux_npu_mnv3_fer_real_probe.csv"
        raw = REPO / "results" / "t527_linux_npu_mnv3_fer_real_probe.log"
        note = (
            "ACUITY uint8 real-calib (FER+/ExpW 128); "
            "VIP9000NANOSI_PLUS_PID0X10000016; latency-only"
        )
        model = "MobileNetV3_Small_FER_uint8_nbg_real"
    else:
        local_nbg = REPO / "deploy" / "t527" / "models" / "mobilenetv3_small_fer_uint8.nb"
        out = REPO / "results" / "t527_linux_npu_mnv3_fer_probe.csv"
        raw = REPO / "results" / "t527_linux_npu_mnv3_fer_probe.log"
        note = (
            "ACUITY uint8 asymmetric_affine; "
            "VIP9000NANOSI_PLUS_PID0X10000016; synthetic calib; latency-only"
        )
        model = "MobileNetV3_Small_FER_uint8_nbg"

    if not local_nbg.exists():
        raise SystemExit(f"missing {local_nbg}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        inp = td_path / "112x112_u8.dat"
        inp.write_bytes(bytes([ZERO_POINT]) * IN_ELEMS)
        sample = td_path / "sample.txt"
        sample.write_text(
            (
                "vpm_run is a framework for testing VIPLite driver. \n"
                "\n"
                "#You may put comment by starting with a '#'.\n"
                "#####################################################A separator makes the list clear.\n"
                "[network]\n"
                f"{REMOTE_DIR}/network_binary.nb\n"
                "[input]\n"
                f"{REMOTE_DIR}/112x112_u8.dat\n"
                "\n"
            ),
            encoding="utf-8",
            newline="\n",
        )

        run(f"mkdir -p {REMOTE_DIR}", HOST)
        put(str(local_nbg), f"{REMOTE_DIR}/network_binary.nb", HOST)
        put(str(inp), f"{REMOTE_DIR}/112x112_u8.dat", HOST)
        put(str(sample), f"{REMOTE_DIR}/sample.txt", HOST)

    code, out_txt, err = run(
        f"chmod 666 /dev/vipcore 2>/dev/null || true; "
        f"cd {REMOTE_DIR} && vpm_run -s sample.txt -l 200 -b 1 2>&1",
        HOST,
        timeout=300,
    )
    text = out_txt + err
    raw.write_text(text, encoding="utf-8")
    if code != 0:
        raise SystemExit(f"vpm_run failed code={code}\n{text[-2000:]}")

    times = [int(x) for x in re.findall(r"profile inference time=(\d+)us", text)]
    if not times:
        raise SystemExit(f"no timings parsed\n{text[-2000:]}")
    use = times[10:] if len(times) > 20 else times
    p50 = statistics.median(use)
    mean = statistics.mean(use)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
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
                "model": model,
                "iters": len(use),
                "latency_us_p50": int(p50),
                "latency_us_mean": int(mean),
                "latency_us_min": min(use),
                "latency_us_max": max(use),
                "note": note,
            }
        )
    print(f"OK n={len(use)} p50_us={int(p50)} mean_us={int(mean)} -> {out}")


if __name__ == "__main__":
    main()
