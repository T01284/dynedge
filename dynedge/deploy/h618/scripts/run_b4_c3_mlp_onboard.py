#!/usr/bin/env python3
"""B-4: Export C3 2-layer ExitGate MLP → NCNN and microbench on H618.

Goal: prove on-board gate cost ≪ 0.1 ms so host-decision (--exits) is not a
material wall-clock cheat vs embedding the MLP.

Usage:
  python deploy/h618/scripts/run_b4_c3_mlp_onboard.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import paramiko
import torch
import torch.nn as nn
from scp import SCPClient

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.latency_controller import ExitGate, LatencyBudgetController  # noqa: E402

HOST, USER, PW = "BOARD_IP_REDACTED", "root", "1234"
CKPT = (
    ROOT
    / "results/from_autodl/formal_v4/controller_band"
    / "ferplus_h618_b22.42ms_l0p2"
    / "controller_ferplus_h618_b22.42ms_l0p2_seed42_v4_band_retry_best.pt"
)
OUT = ROOT / "results/h618_v4/b4_c3_mlp"
REMOTE_DIR = "/root/fer-dynedge/models/c3_mlp_gates"
REMOTE_OUT = "/root/fer-dynedge/results/b4_c3_mlp"
OP_BENCH = "/root/fer-dynedge/deploy/ncnn_edge/build-opbench/fer_op_bench"
THRESHOLD_MS = 0.1


class GateNCHW(nn.Module):
    """Wrap ExitGate so fer_op_bench can feed Mat(w=1,h=1,c=8)."""

    def __init__(self, gate: ExitGate):
        super().__init__()
        self.gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W) with C=8,H=W=1  or (N,8)
        if x.dim() == 4:
            x = x.flatten(1)
        return self.gate(x)


def load_ctrl() -> LatencyBudgetController:
    import pathlib

    try:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]
    except Exception:
        pass
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    ctrl = LatencyBudgetController()
    state = ckpt["controller"] if isinstance(ckpt, dict) and "controller" in ckpt else ckpt
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    ctrl.load_state_dict(state, strict=True)
    ctrl.eval()
    return ctrl


def export_gate(gate: ExitGate, work: Path, stem: str) -> tuple[Path, Path, float]:
    import pnnx

    work.mkdir(parents=True, exist_ok=True)
    wrapped = GateNCHW(gate).eval()
    pt = str(work / "m.pt")
    param = work / "m.ncnn.param"
    binp = work / "m.ncnn.bin"
    xt = torch.randn(1, 8, 1, 1)
    try:
        pnnx.export(
            wrapped,
            pt,
            inputs=xt,
            ncnnparam=str(param),
            ncnnbin=str(binp),
            fp16=False,
        )
    except Exception as e:
        print(f"  pnnx post-err (ok if files exist): {type(e).__name__}: {e}")
    if not param.exists() or not binp.exists():
        raise RuntimeError(f"pnnx failed for {stem}")

    # rename blobs to input/output for fer_op_bench defaults (override via CLI anyway)
    text = param.read_text(encoding="utf-8")
    text = text.replace(" in0", " input").replace("\tin0", "\tinput")
    text = text.replace(" out0", " output").replace("\tout0", "\toutput")
    lines = []
    for line in text.splitlines():
        if line.startswith("Input"):
            line = "Input                    input                     0 1 input"
        lines.append(line)
    param.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # host verify
    import ncnn

    with torch.no_grad():
        zt = wrapped(xt).numpy().reshape(-1)
    net = ncnn.Net()
    net.opt.num_threads = 1
    assert net.load_param(str(param)) == 0
    assert net.load_model(str(binp)) == 0
    mat = ncnn.Mat(1, 1, 8)
    arr = xt.numpy().reshape(8)
    for i in range(8):
        mat.channel(i)[0] = float(arr[i])
    ex = net.create_extractor()
    # try input name
    for in_name in ("input", "in0"):
        try:
            assert ex.input(in_name, mat) == 0
            break
        except Exception:
            continue
    out = None
    for out_name in ("output", "out0", "out"):
        ret, out = ex.extract(out_name)
        if ret == 0:
            break
    assert out is not None and ret == 0
    zn = np.array([out[i] for i in range(out.w)], dtype=np.float32)
    cos = float(np.dot(zt, zn) / (np.linalg.norm(zt) * np.linalg.norm(zn) + 1e-12))
    print(f"  verify {stem} cos={cos:.6f} torch={zt} ncnn={zn}")

    out_param = OUT / f"{stem}.param"
    out_bin = OUT / f"{stem}.bin"
    out_param.write_bytes(param.read_bytes())
    out_bin.write_bytes(binp.read_bytes())
    # keep rewritten text
    out_param.write_text(param.read_text(encoding="utf-8"), encoding="utf-8")
    return out_param, out_bin, cos


def ssh_run(c: paramiko.SSHClient, cmd: str, timeout: int = 120) -> str:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    code = o.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"exit={code}\n{out}\n{err}")
    return out + err


def bench_remote(c: paramiko.SSHClient, stem: str, threads: int = 4) -> dict:
    rparam = f"{REMOTE_DIR}/{stem}.param"
    rbin = f"{REMOTE_DIR}/{stem}.bin"
    log = f"{REMOTE_OUT}/{stem}_t{threads}.txt"
    cmd = (
        f"{OP_BENCH} --device h618 --param {rparam} --bin {rbin} "
        f"--h 1 --w 1 --c 8 --input input --output output "
        f"--threads {threads} --warmup 200 --iters 2000 > {log} 2>&1; "
        f"cat {log}"
    )
    text = ssh_run(c, cmd, timeout=180)
    m = re.search(r"latency_us_p50=([0-9.]+)", text)
    mean = re.search(r"latency_us_mean=([0-9.]+)", text)
    p95 = re.search(r"latency_us_p95=([0-9.]+)", text)
    if not m:
        raise RuntimeError("no RESULT\n" + text[-1500:])
    return {
        "stem": stem,
        "threads": threads,
        "p50_us": float(m.group(1)),
        "mean_us": float(mean.group(1)) if mean else None,
        "p95_us": float(p95.group(1)) if p95 else None,
        "p50_ms": float(m.group(1)) / 1000.0,
        "raw": text.strip().splitlines()[-1],
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    work_root = OUT / "pnnx_work"
    if work_root.exists():
        shutil.rmtree(work_root)

    print(f"load {CKPT}", flush=True)
    ctrl = load_ctrl()
    rows = []
    for i, gate in enumerate(ctrl.gates):
        stem = f"c3_exit_gate{i}_h32"
        print(f"== export {stem} ==", flush=True)
        export_gate(gate, work_root / stem, stem)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=<REDACTED> timeout=20, allow_agent=False, look_for_keys=False)
    ssh_run(c, f"mkdir -p {REMOTE_DIR} {REMOTE_OUT}")
    with SCPClient(c.get_transport()) as scp:
        for i in range(3):
            stem = f"c3_exit_gate{i}_h32"
            scp.put(str(OUT / f"{stem}.param"), f"{REMOTE_DIR}/{stem}.param")
            scp.put(str(OUT / f"{stem}.bin"), f"{REMOTE_DIR}/{stem}.bin")
            print(f"PUT {stem}", flush=True)

    # performance governor
    ssh_run(
        c,
        "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done; "
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
    )

    for threads in (1, 4):
        for i in range(3):
            stem = f"c3_exit_gate{i}_h32"
            print(f"== bench {stem} threads={threads} ==", flush=True)
            r = bench_remote(c, stem, threads=threads)
            rows.append(r)
            print(f"  p50={r['p50_us']:.1f} us ({r['p50_ms']:.4f} ms)", flush=True)

    c.close()

    # summary: worst-case 3 gates sequential (continue all the way)
    by_t = {}
    for r in rows:
        by_t.setdefault(r["threads"], []).append(r)
    summary = {
        "schema": "b4_c3_mlp_onboard_v1",
        "device": "h618",
        "ckpt": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "gate": "ExitGate 8→32→2 ReLU",
        "threshold_ms": THRESHOLD_MS,
        "rows": rows,
        "verdict": {},
    }
    for th, rs in by_t.items():
        p50s = [r["p50_us"] for r in rs]
        one = float(np.median(p50s))
        three = 3.0 * one
        ok = (three / 1000.0) < THRESHOLD_MS
        summary["verdict"][f"threads_{th}"] = {
            "gate_p50_us_median": one,
            "three_gates_us": three,
            "three_gates_ms": three / 1000.0,
            "pass_lt_0p1ms": ok,
        }
        print(
            f"threads={th}: 1gate≈{one:.1f}us  3gates≈{three:.1f}us "
            f"({three/1000:.4f}ms)  PASS={ok}",
            flush=True,
        )

    (OUT / "b4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # also csv
    import csv

    with (OUT / "b4_gate_latency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["stem", "threads", "p50_us", "mean_us", "p95_us", "p50_ms", "raw"],
        )
        w.writeheader()
        w.writerows(rows)

    all_pass = all(v["pass_lt_0p1ms"] for v in summary["verdict"].values())
    print(f"ALL_PASS={all_pass} -> {OUT / 'b4_summary.json'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
