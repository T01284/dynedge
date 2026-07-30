#!/usr/bin/env python3
"""Mine full lut/enum_* h618_ncnn ops (fp32+int8) via fer_op_bench + ncnn2int8.

Resumes from existing CSV; writes remote batch runner for speed.
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lut"))
from topo_ops import op_key  # noqa: E402

HOST = "BOARD_IP_REDACTED"
PASS = "1234"
REMOTE_ROOT = "/root/fer-dynedge"
REMOTE_WORK = f"{REMOTE_ROOT}/lut_ops_full"
REMOTE_EDGE = f"{REMOTE_ROOT}/deploy/ncnn_edge"
REMOTE_BENCH = f"{REMOTE_EDGE}/build-opbench/fer_op_bench"


def _write_f32_layer(bin_f, weights: list[float], bias: list[float] | None = None) -> None:
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


def write_add(param: Path, bin_path: Path, *, cin: int, h: int, w: int) -> None:
    # BinaryOp Add: two inputs same shape — use MemoryData + Input then BinaryOp
    # Simpler: use eltwise with two Inputs via Split — for LUT use AbsVal-ish cost proxy:
    # Unary Abs on one map as lower bound; better: Eltwise sum of input with itself via Split.
    lines = [
        "7767517",
        "3 3",
        f"Input            input           0 1 input 0={w} 1={h} 2={cin}",
        "Split            split0          1 2 input a b",
        "Eltwise          add0            2 1 a b out 0=1",  # 0=1 sum
    ]
    param.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    bin_path.write_bytes(b"")  # no weights


def write_gap(param: Path, bin_path: Path, *, cin: int, h: int, w: int) -> None:
    lines = [
        "7767517",
        "2 2",
        f"Input            input           0 1 input 0={w} 1={h} 2={cin}",
        "Pooling          gap0            1 1 input out 0=1 4=1",  # global avg: pooling_type=1, global=1
    ]
    param.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    bin_path.write_bytes(b"")


def load_done(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    done = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok", "")).lower() in {"1", "true", "True"} and r.get("p50_us"):
                # rebuild key from fields
                o = {k: r.get(k) for k in r}
                for num in ("cin", "cout", "k", "s", "h", "w", "m", "n", "groups", "width"):
                    if o.get(num) not in (None, ""):
                        try:
                            o[num] = float(o[num]) if "." in str(o[num]) else int(float(o[num]))
                        except ValueError:
                            pass
                done[op_key(o)] = r
    return done


def ensure_op_bench(c: paramiko.SSHClient, sftp: paramiko.SFTPClient) -> None:
    for rel in [
        "deploy/ncnn_edge/src/fer_op_bench.cpp",
        "deploy/ncnn_edge/CMakeLists.txt",
    ]:
        sftp.put(str(ROOT / rel), f"{REMOTE_ROOT}/{rel.replace(chr(92), '/')}")
    cmd = (
        f"export NCNN_DIR=/opt/ncnn/install; "
        f"cmake -S {REMOTE_EDGE} -B {REMOTE_EDGE}/build-opbench "
        f"-DFER_HAS_NCNN=ON -DNCNN_DIR=$NCNN_DIR -DFER_NUM_CLASSES=8 >/dev/null && "
        f"cmake --build {REMOTE_EDGE}/build-opbench -j$(nproc) --target fer_op_bench"
    )
    stdin, stdout, stderr = c.exec_command(cmd, timeout=300)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError(f"build failed\n{out}\n{err}")
    print("fer_op_bench ready", flush=True)


def quantize_int8(c: paramiko.SSHClient, stem: str, h: int, w: int, cin: int) -> tuple[str, str] | None:
    """Produce int8 param/bin next to fp32; return paths or None."""
    fp = f"{REMOTE_WORK}/{stem}"
    ip = f"{REMOTE_WORK}/{stem}_int8"
    # synthetic calib: one random ppm-like via python on board
    cmd = f"""
set -e
export PATH=/opt/ncnn/install/bin:$PATH
CALIB={REMOTE_WORK}/calib_{h}x{w}x{cin}.txt
IMG={REMOTE_WORK}/calib_{h}x{w}x{cin}.jpg
python3 - <<'PY'
from pathlib import Path
import struct, zlib
h,w,c={h},{w},{cin}
# write a minimal raw then convert with PIL if available, else skip
try:
    from PIL import Image
    import numpy as np
    arr = (np.random.rand(h,w,min(c,3))*255).astype('uint8')
    if c==1:
        Image.fromarray(arr[:,:,0], 'L').convert('RGB').save('{REMOTE_WORK}/calib_{h}x{w}x{cin}.jpg')
    else:
        Image.fromarray(arr[:,:,:3], 'RGB').save('{REMOTE_WORK}/calib_{h}x{w}x{cin}.jpg')
except Exception as e:
    # fallback: copy any existing jpg
    import shutil, glob
    xs=glob.glob('/root/fer-dynedge/data/calib/**/*.jpg', recursive=True)
    if not xs:
        raise SystemExit('no calib image')
    shutil.copy(xs[0], '{REMOTE_WORK}/calib_{h}x{w}x{cin}.jpg')
Path('{REMOTE_WORK}/calib_{h}x{w}x{cin}.txt').write_text('{REMOTE_WORK}/calib_{h}x{w}x{cin}.jpg\\n')
print('calib ok')
PY
rm -f {fp}.table {ip}.param {ip}.bin
ncnn2table {fp}.param {fp}.bin $CALIB {fp}.table \\
  mean=[0,0,0] norm=[0.00392157,0.00392157,0.00392157] shape=[{h},{w},{cin}] pixel=RGB thread=2 method=kl \\
  || ncnn2table {fp}.param {fp}.bin $CALIB {fp}.table \\
  mean=[0,0,0] norm=[1,1,1] shape=[{h},{w},{cin}] pixel=RGB thread=1 method=kl
ncnn2int8 {fp}.param {fp}.bin {ip}.param {ip}.bin {fp}.table
test -s {ip}.param && test -s {ip}.bin
"""
    stdin, stdout, stderr = c.exec_command(cmd, timeout=180)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        return None
    return f"{ip}.param", f"{ip}.bin"


def bench_one(c: paramiko.SSHClient, param: str, binp: str, h: int, w: int, ch: int) -> float | None:
    cmd = (
        f"{REMOTE_BENCH} --device h618 --param {param} --bin {binp} "
        f"--h {h} --w {w} --c {ch} --threads 4 --warmup 15 --iters 100"
    )
    stdin, stdout, stderr = c.exec_command(cmd, timeout=120)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    for line in out.splitlines():
        if line.startswith("RESULT"):
            for part in line.split():
                if part.startswith("latency_us_p50="):
                    return float(part.split("=", 1)[1])
    if err.strip():
        print(err[-400:], flush=True)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum", type=Path, default=ROOT / "lut/enum_v4_full.json")
    ap.add_argument("--out", type=Path, default=ROOT / "lut/ops_h618_v4_full.csv")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--ops", default="conv,dwconv,gemm,add,gap")
    ap.add_argument("--dtype", default="", help="fp32|int8|empty=both")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    want_ops = set(args.ops.split(","))
    enum = json.loads(args.enum.read_text(encoding="utf-8"))
    ops = [
        o
        for o in enum["ops"]
        if o.get("backend") == "h618_ncnn"
        and o.get("op") in want_ops
        and (not args.dtype or o.get("dtype") == args.dtype)
    ]
    # unique by op_key
    seen = set()
    uniq = []
    for o in ops:
        k = op_key(o)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(o)
    if args.limit:
        uniq = uniq[: args.limit]
    print(f"to mine: {len(uniq)}", flush=True)

    done = load_done(args.out)
    print(f"resume done: {len(done)}", flush=True)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="root", password=<REDACTED> timeout=30, allow_agent=False, look_for_keys=False)
    sftp = c.open_sftp()
    if not args.skip_build:
        ensure_op_bench(c, sftp)
    try:
        sftp.mkdir(REMOTE_WORK)
    except OSError:
        pass

    local_dir = ROOT / "lut" / "_ops_full_tmp"
    local_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = None
    rows_out: list[dict] = []
    # keep previous good rows
    if args.out.exists():
        with args.out.open(encoding="utf-8", newline="") as f:
            rows_out = list(csv.DictReader(f))
            if rows_out:
                fieldnames = list(rows_out[0].keys())

    n_ok = n_skip = n_fail = 0
    for i, o in enumerate(uniq):
        k = op_key(o)
        if k in done:
            n_skip += 1
            continue
        name = f"op_{i:04d}_{o['op']}_{o.get('dtype')}"
        param_l = local_dir / f"{name}.param"
        bin_l = local_dir / f"{name}.bin"
        try:
            if o["op"] in {"conv", "dwconv"}:
                cin = int(o["cin"])
                cout = int(o["cout"])
                kk = int(o["k"])
                s = int(o["s"])
                groups = int(o.get("groups") or (cin if o["op"] == "dwconv" else 1))
                h = int(o["h"])
                w = int(o.get("w") or h)
                write_conv(param_l, bin_l, cin=cin, cout=cout, k=kk, s=s, groups=groups, h=h, w=w)
                ih, iw, ic = h, w, cin
            elif o["op"] == "gemm":
                kk = int(o["k"])
                nn = int(o["n"])
                write_gemm(param_l, bin_l, k=kk, n=nn)
                ih, iw, ic = 1, kk, 1
            elif o["op"] == "add":
                cin = int(o["cin"])
                h = int(o["h"])
                w = int(o.get("w") or h)
                write_add(param_l, bin_l, cin=cin, h=h, w=w)
                ih, iw, ic = h, w, cin
            elif o["op"] == "gap":
                cin = int(o["cin"])
                h = int(o["h"])
                w = int(o.get("w") or h)
                write_gap(param_l, bin_l, cin=cin, h=h, w=w)
                ih, iw, ic = h, w, cin
            else:
                continue
        except Exception as e:
            print(f"[{i+1}/{len(uniq)}] GEN FAIL {o.get('note')} {e}", flush=True)
            n_fail += 1
            continue

        rem_p = f"{REMOTE_WORK}/{name}.param"
        rem_b = f"{REMOTE_WORK}/{name}.bin"
        sftp.put(str(param_l), rem_p)
        sftp.put(str(bin_l), rem_b)

        lat = None
        if o.get("dtype") == "int8":
            q = quantize_int8(c, name, ih, iw, ic)
            if q:
                lat = bench_one(c, q[0], q[1], ih, iw, ic)
        else:
            lat = bench_one(c, rem_p, rem_b, ih, iw, ic)

        row = {**o, "p50_us": lat if lat is not None else "", "ok": bool(lat is not None)}
        # update / append
        rows_out = [r for r in rows_out if op_key({**r, **{kk: r.get(kk) for kk in r}}) != k]
        # simpler: append and rewrite unique at end
        rows_out.append({str(kk): ("" if vv is None else vv) for kk, vv in row.items()})
        if lat is not None:
            n_ok += 1
            done[k] = row
        else:
            n_fail += 1
        print(
            f"[{i+1}/{len(uniq)}] {o['op']} {o.get('dtype')} {o.get('note')} "
            f"w={o.get('width')} h={o.get('h') or o.get('k')} p50={lat}",
            flush=True,
        )

        # periodic save
        if (n_ok + n_fail) % 10 == 0:
            _save_csv(args.out, rows_out)

    _save_csv(args.out, rows_out)
    sftp.close()
    c.close()
    print(f"wrote {args.out} ok={n_ok} skip={n_skip} fail={n_fail} total_rows={len(rows_out)}")
    return 0 if n_fail == 0 or n_ok > 0 else 1


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # dedupe by op_key keeping last
    best: dict[tuple, dict] = {}
    for r in rows:
        rr = dict(r)
        for num in ("cin", "cout", "k", "s", "h", "w", "m", "n", "groups", "width", "p50_us"):
            if rr.get(num) not in (None, ""):
                try:
                    rr[num] = float(rr[num]) if "." in str(rr[num]) else int(float(rr[num]))
                except (ValueError, TypeError):
                    pass
        best[op_key(rr)] = r
    out_rows = list(best.values())
    keys = sorted({k for r in out_rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    raise SystemExit(main())
