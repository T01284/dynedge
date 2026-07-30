#!/usr/bin/env python3
"""Mine a subset of lut/enum_*.json on H618 via tiny single-op NCNN nets + fer_op_bench."""
from __future__ import annotations

import argparse
import csv
import json
import struct
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "BOARD_IP_REDACTED"
PASS = "1234"
REMOTE_ROOT = "/root/fer-dynedge"
REMOTE_WORK = f"{REMOTE_ROOT}/lut_ops"
REMOTE_EDGE = f"{REMOTE_ROOT}/deploy/ncnn_edge"
REMOTE_BENCH = f"{REMOTE_EDGE}/build-opbench/fer_op_bench"


def _write_f32_layer(bin_f, weights: list[float], bias: list[float] | None = None) -> None:
    # Modern ncnn ModelBin: leading int32 flag=0 means raw float32 weights.
    bin_f.write(struct.pack("<i", 0))
    for v in weights:
        bin_f.write(struct.pack("<f", float(v)))
    if bias is not None:
        for v in bias:
            bin_f.write(struct.pack("<f", float(v)))


def write_conv(param: Path, bin_path: Path, *, cin: int, cout: int, k: int, s: int, groups: int, h: int, w: int) -> None:
    pad = k // 2
    if groups > 1 and groups == cin == cout:
        n_w = cout * k * k
        # Match onnx2ncnn: set both H/W kernel/stride/pad + groups (param 7).
        layer = (
            f"ConvolutionDepthWise conv0 1 1 input out "
            f"0={cout} 1={k} 11={k} 2=1 12=1 3={s} 13={s} "
            f"4={pad} 14={pad} 15={pad} 16={pad} 5=1 6={n_w} 7={groups}"
        )
    else:
        n_w = cin * cout * k * k // max(groups, 1)
        layer = (
            f"Convolution      conv0          1 1 input out "
            f"0={cout} 1={k} 11={k} 2=1 12=1 3={s} 13={s} "
            f"4={pad} 14={pad} 15={pad} 16={pad} 5=1 6={n_w}"
        )
    lines = [
        "7767517",
        "2 2",
        f"Input            input           0 1 input 0={w} 1={h} 2={cin}",
        layer,
    ]
    param.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with bin_path.open("wb") as f:
        _write_f32_layer(f, [0.01] * n_w, [0.0] * cout)


def write_gemm(param: Path, bin_path: Path, *, k: int, n: int) -> None:
    lines = [
        "7767517",
        "2 2",
        f"Input            input           0 1 input 0={k} 1=1 2=1",
        f"InnerProduct     fc0             1 1 input out 0={n} 1=1 2={k * n}",
    ]
    param.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with bin_path.open("wb") as f:
        _write_f32_layer(f, [0.01] * (k * n), [0.0] * n)


def ensure_op_bench(c: paramiko.SSHClient, sftp: paramiko.SFTPClient) -> None:
    local_cpp = ROOT / "deploy/ncnn_edge/src/fer_op_bench.cpp"
    rem_cpp = f"{REMOTE_EDGE}/src/fer_op_bench.cpp"
    sftp.put(str(local_cpp), rem_cpp)
    # also sync CMakeLists
    sftp.put(str(ROOT / "deploy/ncnn_edge/CMakeLists.txt"), f"{REMOTE_EDGE}/CMakeLists.txt")
    cmd = (
        f"export NCNN_DIR=/opt/ncnn/install; "
        f"cmake -S {REMOTE_EDGE} -B {REMOTE_EDGE}/build-opbench "
        f"-DFER_HAS_NCNN=ON -DNCNN_DIR=$NCNN_DIR -DFER_NUM_CLASSES=8 >/dev/null && "
        f"cmake --build {REMOTE_EDGE}/build-opbench -j$(nproc) --target fer_op_bench"
    )
    stdin, stdout, stderr = c.exec_command(cmd, timeout=300)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f"build fer_op_bench failed rc={rc}\n{out}\n{err}")
    print("fer_op_bench ready", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum", type=Path, default=ROOT / "lut/enum_v4_firstpass.json")
    ap.add_argument("--out", type=Path, default=ROOT / "lut/ops_h618_v4_sample.csv")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--backend-filter", default="h618_ncnn")
    ap.add_argument("--dtype-filter", default="fp32")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--wait-idle", action="store_true", help="wait until board_v4_matrix exits")
    args = ap.parse_args()

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="root", password=<REDACTED> timeout=30, allow_agent=False, look_for_keys=False)

    if args.wait_idle:
        print("waiting for V4 MATRIX DONE...", flush=True)
        while True:
            stdin, stdout, _ = c.exec_command(
                "grep -c 'V4 MATRIX DONE' /root/fer-dynedge/results/overnight_v4/v4_status.txt 2>/dev/null || echo 0"
            )
            n = (stdout.read().decode().strip().splitlines() or ["0"])[-1].strip()
            if n.isdigit() and int(n) >= 1:
                break
            print(f"  not done yet (grep={n})", flush=True)
            time.sleep(60)

    sftp = c.open_sftp()
    if not args.skip_build:
        ensure_op_bench(c, sftp)
    try:
        sftp.mkdir(REMOTE_WORK)
    except OSError:
        pass

    enum = json.loads(args.enum.read_text(encoding="utf-8"))
    ops = [
        o
        for o in enum["ops"]
        if o.get("backend") == args.backend_filter
        and o.get("dtype") == args.dtype_filter
        and o.get("op") in {"conv", "dwconv", "gemm"}
    ]
    seen = set()
    uniq = []
    for o in ops:
        key = (
            o["op"],
            o.get("cin"),
            o.get("cout"),
            o.get("k"),
            o.get("s"),
            o.get("h"),
            o.get("w"),
            o.get("m"),
            o.get("n"),
            o.get("groups"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(o)
    uniq = uniq[: args.limit]

    local_dir = ROOT / "lut" / "_ops_tmp"
    local_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, o in enumerate(uniq):
        name = f"op_{i:03d}_{o['op']}"
        param_l = local_dir / f"{name}.param"
        bin_l = local_dir / f"{name}.bin"
        if o["op"] in {"conv", "dwconv"}:
            cin = int(o["cin"])
            cout = int(o["cout"])
            k = int(o["k"])
            s = int(o["s"])
            groups = int(o.get("groups") or (cin if o["op"] == "dwconv" else 1))
            h = int(o["h"])
            w = int(o.get("w") or h)
            write_conv(param_l, bin_l, cin=cin, cout=cout, k=k, s=s, groups=groups, h=h, w=w)
            ih, iw, ic = h, w, cin
        else:
            kk = int(o["k"])
            nn = int(o["n"])
            write_gemm(param_l, bin_l, k=kk, n=nn)
            ih, iw, ic = 1, kk, 1

        rem_p = f"{REMOTE_WORK}/{name}.param"
        rem_b = f"{REMOTE_WORK}/{name}.bin"
        sftp.put(str(param_l), rem_p)
        sftp.put(str(bin_l), rem_b)
        cmd = (
            f"{REMOTE_BENCH} --device h618 --param {rem_p} --bin {rem_b} "
            f"--h {ih} --w {iw} --c {ic} --threads 4 --warmup 20 --iters 200"
        )
        stdin, stdout, stderr = c.exec_command(cmd, timeout=120)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        lat = None
        for line in out.splitlines():
            if line.startswith("RESULT"):
                for part in line.split():
                    if part.startswith("latency_us_p50="):
                        lat = float(part.split("=", 1)[1])
        row = {**o, "p50_us": lat, "ok": lat is not None}
        rows.append(row)
        print(f"[{i+1}/{len(uniq)}] {o['op']} {o.get('note')} p50={lat}", flush=True)
        if lat is None:
            print((out or err)[-800:])

    sftp.close()
    c.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with args.out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"wrote {args.out} ok={ok}/{len(rows)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
