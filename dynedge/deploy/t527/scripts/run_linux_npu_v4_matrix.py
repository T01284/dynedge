#!/usr/bin/env python3
"""T527: v4 Pareto/MCU anchor NBG → vpm_run latency matrix (NPU e2e).

Mirrors run_linux_npu_mnv3_fer.py for variable HxW (64/96/112).
Latency-only: fills u8 input with constant zp (default 128) of correct NCHW size.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import tempfile
import time
from pathlib import Path

from ssh_t527 import put, run

HOST = "BOARD_IP_REDACTED"
REPO = Path(__file__).resolve().parents[3]
NBG_DIR = REPO / "deploy" / "t527" / "models" / "v4_npu"
OUT_DIR = REPO / "results" / "t527_v4"
REMOTE_BASE = "/root/fer-dynedge/npu/v4"


def hw_of(stem: str) -> int:
    if "img64" in stem or "_s64g_" in stem:
        return 64
    if "img96" in stem or "_s96c_" in stem:
        return 96
    return 112


def short_name(stem: str) -> str:
    # avoid long paths / PowerShell \r issues on remote
    m = re.search(r"w[\d.]+_d\d+_s\d+[gc]", stem)
    return m.group(0).replace(".", "") if m else stem[-24:]


def make_sample(remote_dir: str, dat_name: str) -> str:
    return (
        "vpm_run is a framework for testing VIPLite driver. \n"
        "\n"
        "#You may put comment by starting with a '#'.\n"
        "#####################################################A separator makes the list clear.\n"
        "[network]\n"
        f"{remote_dir}/network_binary.nb\n"
        "[input]\n"
        f"{remote_dir}/{dat_name}\n"
        "\n"
    )


def parse_times(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"profile inference time=(\d+)us", text)]


def bench_one(nbg: Path, loops: int, zp: int) -> dict:
    stem = nbg.stem.replace("_uint8", "")
    hw = hw_of(stem)
    tag = short_name(stem)
    remote = f"{REMOTE_BASE}/{tag}"
    dat = f"{hw}x{hw}_u8.dat"
    elems = 1 * 3 * hw * hw

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        inp = td_path / dat
        inp.write_bytes(bytes([zp & 0xFF]) * elems)
        sample = td_path / "sample.txt"
        sample.write_text(make_sample(remote, dat), encoding="utf-8", newline="\n")

        run(f"mkdir -p {remote}", HOST)
        put(str(nbg), f"{remote}/network_binary.nb", HOST)
        put(str(inp), f"{remote}/{dat}", HOST)
        put(str(sample), f"{remote}/sample.txt", HOST)

    code, out_txt, err = run(
        f"chmod 666 /dev/vipcore 2>/dev/null || true; "
        f"cd {remote} && vpm_run -s sample.txt -l {loops} -b 1 2>&1",
        HOST,
        timeout=600,
    )
    text = out_txt + err
    log = OUT_DIR / f"npu_{stem}.log"
    log.write_text(text, encoding="utf-8")
    if code != 0:
        raise RuntimeError(f"vpm_run exit={code}\n{text[-2500:]}")

    times = parse_times(text)
    if not times:
        raise RuntimeError(f"no timings\n{text[-2500:]}")
    use = times[10:] if len(times) > 20 else times
    return {
        "model": stem,
        "tag": tag,
        "input": f"{hw}x{hw}x3",
        "zp": zp,
        "loops_parsed": len(times),
        "iters_used": len(use),
        "latency_us_p50": int(statistics.median(use)),
        "latency_us_mean": int(statistics.mean(use)),
        "latency_us_min": int(min(use)),
        "latency_us_max": int(max(use)),
        "backend": "viplite_npu",
        "device": "t527",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=200)
    ap.add_argument("--zp", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = OUT_DIR / "npu_matrix_summary.csv"
    status = OUT_DIR / "npu_status.txt"

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

    # pin CPU for fair compare with NCNN matrix
    run(
        "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done",
        HOST,
    )

    with status.open("a", encoding="utf-8") as sf:
        sf.write(f"===== NPU V4 MATRIX START {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n")
        sf.flush()
        for nbg in nbgs:
            stem = nbg.stem.replace("_uint8", "")
            if stem in done:
                print(f"SKIP {stem}", flush=True)
                continue
            print(f"== NPU {stem} ==", flush=True)
            try:
                row = bench_one(nbg, args.loops, args.zp)
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
            print(
                f"OK {stem} p50={row['latency_us_p50']}us ({row['input']})",
                flush=True,
            )
            sf.write(f"OK {stem} p50={row['latency_us_p50']}\n")
            sf.flush()
        sf.write(
            f"===== NPU V4 MATRIX END {time.strftime('%Y-%m-%dT%H:%M:%S')} "
            f"rows={len(rows)} =====\n"
        )

    print(f"DONE -> {summary} ({len(rows)} rows)")
    return 0 if len(rows) == len(nbgs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
