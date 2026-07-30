#!/usr/bin/env python3
"""Mine MCU ESP-DL op LUT: single-op ONNX → PTQ → embed → flash → CMD BENCH_SYN.

Covers enum esp32p4/esp32s3 conv+dwconv (+gemm as ONNX Gemm when feasible).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import serial
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lut"))
from topo_ops import op_key  # noqa: E402

FW = ROOT / "deploy/mcu_espdl/firmware"
MODELS = ROOT / "deploy/mcu_espdl/models/ops"
OUT_DEFAULT = ROOT / "lut/ops_espdl_v4_firstpass.csv"


class ConvOp(nn.Module):
    def __init__(self, cin, cout, k, s, groups):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(cin, cout, k, s, pad, groups=groups, bias=True)

    def forward(self, x):
        return self.conv(x)


class GemmOp(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.fc = nn.Linear(k, n, bias=True)

    def forward(self, x):
        # x: NCHW with H=W=1, C=k OR N×k
        if x.dim() == 4:
            x = x.flatten(1)
        return self.fc(x)


def export_onnx(o: dict, path: Path) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if o["op"] in {"conv", "dwconv"}:
        cin, cout = int(o["cin"]), int(o["cout"])
        k, s = int(o["k"]), int(o["s"])
        groups = int(o.get("groups") or (cin if o["op"] == "dwconv" else 1))
        h = int(o["h"])
        w = int(o.get("w") or h)
        m = ConvOp(cin, cout, k, s, groups).eval()
        dummy = torch.randn(1, cin, h, w)
        torch.onnx.export(
            m,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["out"],
            opset_version=13,
            dynamo=False,
        )
        return h, w, cin
    if o["op"] == "gemm":
        kk, nn_ = int(o["k"]), int(o["n"])
        # Represent FC as 1x1 conv so ESP-PPQ/ESP-DL see NCHW (cin=k,h=1,w=1).
        m = ConvOp(kk, nn_, k=1, s=1, groups=1).eval()
        dummy = torch.randn(1, kk, 1, 1)
        torch.onnx.export(
            m,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["out"],
            opset_version=13,
            dynamo=False,
        )
        return 1, 1, kk
    raise ValueError(o["op"])


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def ptq(onnx: Path, target: str, name: str, cin: int, h: int, w: int) -> Path:
    out_dir = MODELS / target
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(ROOT / "tools/espdl_ptq_op.py"),
            "--onnx",
            str(onnx),
            "--target",
            target,
            "--out-dir",
            str(out_dir),
            "--name",
            name,
            "--cin",
            str(cin),
            "--h",
            str(h),
            "--w",
            str(w),
            "--calib-steps",
            "16",
        ]
    )
    espdl = out_dir / f"{name}_{target}.espdl"
    if not espdl.exists():
        cands = list(out_dir.glob(f"{name}*.espdl"))
        if not cands:
            raise FileNotFoundError(espdl)
        espdl = cands[0]
    return espdl


def flash_and_bench(espdl: Path, target: str, port: str) -> float | None:
    run([sys.executable, str(ROOT / "tools/espdl_embed_model.py"), "--espdl", str(espdl)])
    env = "esp32p4_espdl" if target == "esp32p4" else "esp32s3_espdl"
    pio = FW / "pio.cmd"
    if not pio.exists():
        pio = Path("pio")
        cmd = ["pio", "run", "-e", env, "-t", "upload", "--upload-port", port]
    else:
        cmd = [str(pio), "run", "-e", env, "-t", "upload", "--upload-port", port]
    run(cmd, cwd=FW)
    time.sleep(2.0)
    ser = serial.Serial(port, 115200, timeout=2)
    time.sleep(1.5)
    ser.reset_input_buffer()
    ser.write(b"CMD INFO\n")
    ser.flush()
    t0 = time.time()
    while time.time() - t0 < 8:
        line = ser.readline().decode("utf-8", "replace").rstrip()
        if line:
            print(line)
            if "READY" in line:
                break
    ser.write(b"CMD BENCH_SYN\n")
    ser.flush()
    lats = []
    t0 = time.time()
    while time.time() - t0 < 30:
        line = ser.readline().decode("utf-8", "replace").rstrip()
        if not line:
            continue
        print(line)
        if line.startswith("RESULT") and "latency_us=" in line:
            for part in line.split():
                if part.startswith("latency_us="):
                    lats.append(float(part.split("=", 1)[1]))
        if line.startswith("BENCH_DONE"):
            break
    ser.close()
    if not lats:
        return None
    return float(statistics.median(lats))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum", type=Path, default=ROOT / "lut/enum_v4_full.json")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--targets", default="esp32p4,esp32s3")
    ap.add_argument("--ports", default="SERIAL_PORT,SERIAL_PORT")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ops", default="conv,dwconv")
    ap.add_argument("--skip-flash", action="store_true")
    args = ap.parse_args()

    targets = args.targets.split(",")
    ports = args.ports.split(",")
    port_map = dict(zip(targets, ports))
    want = set(args.ops.split(","))

    enum = json.loads(args.enum.read_text(encoding="utf-8"))
    ops = []
    seen = set()
    for o in enum["ops"]:
        if o.get("backend") not in targets:
            continue
        if o.get("op") not in want:
            continue
        k = op_key(o)
        if k in seen:
            continue
        seen.add(k)
        ops.append(o)
    if args.limit:
        ops = ops[: args.limit]
    print(f"MCU ops to mine: {len(ops)}", flush=True)

    done = {}
    if args.out.exists():
        with args.out.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if str(r.get("ok")).lower() in {"1", "true"} and r.get("p50_us"):
                    oo = dict(r)
                    for num in ("cin", "cout", "k", "s", "h", "w", "m", "n", "groups", "width"):
                        if oo.get(num) not in (None, ""):
                            try:
                                oo[num] = int(float(oo[num]))
                            except ValueError:
                                pass
                    done[op_key(oo)] = r

    rows = []
    if args.out.exists():
        with args.out.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    for i, o in enumerate(ops):
        k = op_key(o)
        if k in done:
            print(f"[{i+1}/{len(ops)}] SKIP {o['backend']} {o.get('note')}", flush=True)
            continue
        target = o["backend"]
        port = port_map[target]
        name = f"op_{i:03d}_{o['op']}_{o.get('note','')}"
        onnx = MODELS / "onnx" / f"{name}.onnx"
        try:
            h, w, c = export_onnx(o, onnx)
        except Exception as e:
            print(f"export fail {name}: {e}", flush=True)
            continue
        try:
            espdl = ptq(onnx, target, name, cin=c, h=h, w=w)
        except Exception as e:
            print(f"ptq fail {name}: {e}", flush=True)
            continue
        lat = None
        if not args.skip_flash:
            try:
                lat = flash_and_bench(espdl, target, port)
            except Exception as e:
                print(f"flash/bench fail {name}: {e}", flush=True)
        row = {**o, "p50_us": lat if lat is not None else "", "ok": lat is not None}
        rows.append({kk: ("" if vv is None else vv) for kk, vv in row.items()})
        if lat is not None:
            done[k] = row
        print(f"[{i+1}/{len(ops)}] {target} {o['op']} {o.get('note')} p50={lat}", flush=True)
        # save
        keys = sorted({kk for r in rows for kk in r.keys()})
        with args.out.open("w", encoding="utf-8", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=keys)
            wri.writeheader()
            wri.writerows(rows)

    nok = sum(1 for r in rows if str(r.get("ok")).lower() in {"1", "true", "True"})
    print(f"wrote {args.out} ok={nok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
