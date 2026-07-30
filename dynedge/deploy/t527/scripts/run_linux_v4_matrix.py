#!/usr/bin/env python3
"""T527 Linux: v4 subnet overnight matrix (copy H618 board_v4_matrix, host-convert + SSH bench).

Prereq:
  - deploy/t527/scripts/prep_v4_ncnn_models.py  (ONNX→ncnn fp32/int8 on WSL)
  - deploy/ncnn_edge/build-linux-aarch64-c{7,8}/fer_ncnn_bench
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import paramiko
from scp import SCPClient

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
OUT = ROOT / "results" / "t527_v4"
BENCH7 = ROOT / "deploy" / "ncnn_edge" / "build-linux-aarch64-c7" / "fer_ncnn_bench"
BENCH8 = ROOT / "deploy" / "ncnn_edge" / "build-linux-aarch64-c8" / "fer_ncnn_bench"

HOST = "BOARD_IP_REDACTED"
REMOTE = "/root/fer-dynedge"
REMOTE_V4 = f"{REMOTE}/models/v4_predistill"
REMOTE_OUT = f"{REMOTE}/results/t527_v4"


def hwc(stem: str) -> tuple[int, int, int]:
    if "img64" in stem or "_s64g_" in stem:
        return 64, 64, 3
    if "img96" in stem or "_s96c_" in stem:
        return 96, 96, 3
    return 112, 112, 3


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="root", password=<REDACTED> timeout=15, allow_agent=False, look_for_keys=False)
    return c


def run(c, cmd: str, timeout: int = 3600) -> str:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    text = o.read().decode("utf-8", "replace") + e.read().decode("utf-8", "replace")
    code = o.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"exit={code}\n{text[-3000:]}")
    return text


def parse_result(text: str) -> dict:
    m = re.search(r"^RESULT\s+(.*)$", text, re.M)
    if not m:
        raise RuntimeError("no RESULT\n" + text[-2000:])
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def put_file(c, local: Path, remote: str) -> None:
    with SCPClient(c.get_transport()) as scp:
        scp.put(str(local), remote)


def ensure_remote(c) -> None:
    run(c, f"mkdir -p {REMOTE}/bin {REMOTE_V4} {REMOTE_OUT}")
    run(
        c,
        "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
    )
    for ncls, local in ((7, BENCH7), (8, BENCH8)):
        if not local.exists():
            raise FileNotFoundError(local)
        put_file(c, local, f"{REMOTE}/bin/fer_ncnn_bench_c{ncls}")
        run(c, f"chmod +x {REMOTE}/bin/fer_ncnn_bench_c{ncls}")


def upload_model(c, stem: str, dtype: str) -> tuple[str, str]:
    if dtype == "fp32":
        param, binp = MODELS / f"{stem}.param", MODELS / f"{stem}.bin"
    else:
        param, binp = MODELS / f"{stem}_int8.param", MODELS / f"{stem}_int8.bin"
    if not param.exists() or not binp.exists():
        raise FileNotFoundError(f"missing {param} / {binp}")
    rparam = f"{REMOTE_V4}/{param.name}"
    rbin = f"{REMOTE_V4}/{binp.name}"
    # skip if remote size matches
    try:
        remote_sz = run(c, f"stat -c %s {rparam} 2>/dev/null || echo 0").strip()
        if remote_sz.isdigit() and int(remote_sz) == param.stat().st_size:
            return rparam, rbin
    except Exception:
        pass
    put_file(c, param, rparam)
    put_file(c, binp, rbin)
    return rparam, rbin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", choices=("both", "fp32", "int8"), default="both")
    ap.add_argument("--skip-upload-bin", action="store_true")
    ap.add_argument("--models", type=Path, default=None, help="override MODELS dir")
    ap.add_argument("--out", type=Path, default=None, help="override OUT dir")
    ap.add_argument("--remote-v4", type=str, default="", help="override REMOTE_V4")
    args = ap.parse_args()

    global MODELS, OUT, REMOTE_V4, REMOTE_OUT
    if args.models:
        MODELS = args.models
    if args.out:
        OUT = args.out
        REMOTE_OUT = f"{REMOTE}/results/{OUT.name}"
    if args.remote_v4:
        REMOTE_V4 = args.remote_v4
    elif args.models:
        REMOTE_V4 = f"{REMOTE}/models/{MODELS.name}"

    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "v4_status.txt"
    logf = OUT / "v4_matrix.log"
    summary = OUT / "matrix_summary.csv"

    stems = sorted({p.stem.replace("_int8", "") for p in MODELS.glob("*.param") if not p.name.endswith("_int8.param")})
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        print(f"ERR no models in {MODELS}")
        return 1

    dtypes = ["fp32", "int8"] if args.dtype == "both" else [args.dtype]
    rows: list[dict] = []
    if summary.exists():
        with summary.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {(r["model"], r["dtype"]) for r in rows}

    c = connect()
    try:
        if not args.skip_upload_bin:
            ensure_remote(c)
        with status.open("a", encoding="utf-8") as sf, logf.open("a", encoding="utf-8") as lf:
            sf.write(f"===== V4 MATRIX START {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n")
            sf.flush()
            for stem in stems:
                h, w, ch = hwc(stem)
                bench = f"{REMOTE}/bin/fer_ncnn_bench_c7" if "expw" in stem else f"{REMOTE}/bin/fer_ncnn_bench_c8"
                for dtype in dtypes:
                    if (stem, dtype) in done:
                        print(f"SKIP {stem} {dtype}", flush=True)
                        continue
                    local_txt = OUT / f"{stem}_{dtype}.txt"
                    if local_txt.exists() and "RESULT" in local_txt.read_text(encoding="utf-8", errors="replace"):
                        text = local_txt.read_text(encoding="utf-8", errors="replace")
                    else:
                        print(f"== {stem} {dtype} {h}x{w} ==", flush=True)
                        try:
                            rparam, rbin = upload_model(c, stem, dtype)
                            cmd = (
                                f"{bench} --device t527 --param {rparam} --bin {rbin} "
                                f"--h {h} --w {w} --c {ch} "
                                f"--threads {args.threads} --warmup {args.warmup} --iters {args.iters}"
                            )
                            text = run(c, cmd, timeout=3600)
                        except Exception as ex:
                            msg = f"FAIL {stem} {dtype}: {ex}"
                            print(msg, flush=True)
                            sf.write(msg + "\n")
                            sf.flush()
                            lf.write(msg + "\n")
                            continue
                        local_txt.write_text(text, encoding="utf-8")
                        # also keep a copy on board
                        run(c, f"cat > {REMOTE_OUT}/{stem}_{dtype}.txt << 'EOF'\n{text}\nEOF")

                    lf.write(f"##### {stem} {dtype}\n{text}\n")
                    lf.flush()
                    d = parse_result(text)
                    row = {
                        "model": stem,
                        "dtype": dtype,
                        "latency_us_p50": d.get("latency_us_p50"),
                        "latency_us_mean": d.get("latency_us_mean"),
                        "latency_us_p95": d.get("latency_us_p95"),
                        "input": f"{h}x{w}x{ch}",
                        "threads": args.threads,
                        "raw": " ".join(f"{k}={v}" for k, v in d.items()),
                    }
                    rows.append(row)
                    done.add((stem, dtype))
                    with summary.open("w", newline="", encoding="utf-8") as f:
                        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        wtr.writeheader()
                        wtr.writerows(rows)
                    print(f"OK {stem} {dtype} p50={row['latency_us_p50']}", flush=True)
                    sf.write(f"OK {stem} {dtype} p50={row['latency_us_p50']}\n")
                    sf.flush()
            sf.write(f"===== V4 MATRIX END {time.strftime('%Y-%m-%dT%H:%M:%S')} rows={len(rows)} =====\n")
    finally:
        c.close()

    print(f"DONE -> {summary} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
