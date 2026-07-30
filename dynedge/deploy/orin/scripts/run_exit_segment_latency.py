#!/usr/bin/env python3
"""Build truncated multi-exit ONNX (exit2/3/4/final) + Orin TRT FP16 segment latency.

Full multi-exit graph always runs to final; segment wall-clock needs truncated engines.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy/orin/scripts"))

from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402
from tools.export_subnet_onnx import remap_v1_stem_keys  # noqa: E402
from ssh_orin import put, run  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "t3ee", ROOT / "deploy/t527/scripts/t3_early_exit_threshold_demo.py"
)
_t3 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_t3)

CKPT = (
    ROOT
    / "results/from_autodl/subnets_v4_predistill"
    / "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_best.pt"
)
OUT = ROOT / "deploy/orin/models/multi_exit"
REMOTE = "/path/to/board/project"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
BENCH = "/path/to/board/project"
RESULT = ROOT / "results/orin_v4/exit_segment_latency"


class Trunc(nn.Module):
    def __init__(self, full: nn.Module, which: str):
        super().__init__()
        self.full = full
        self.which = which  # e2|e3|e4|final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e2, e3, e4, ef = self.full(x)
        return {"e2": e2, "e3": e3, "e4": e4, "final": ef}[self.which]


def load_full() -> nn.Module:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = remap_v1_stem_keys(ckpt["model"])
    ncls = int(state["final_fc.weight"].shape[0])
    slim = ElasticFERSuperNet(num_classes=ncls, arch="v3_wide", slimmable=True)
    slim.load_state_dict(state, strict=True)
    slim.eval()
    cfg = SubnetConfig(width=0.5, depths=(3, 3, 3, 3), img_size=112, gray=False)
    return _t3.BakedMultiExit(slim, cfg).eval()


def export_truncated() -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    full = load_full()
    dummy = torch.randn(1, 3, 112, 112)
    paths = {}
    for which in ("e2", "e3", "e4", "final"):
        onnx = OUT / f"ferplus_w05_d3333_s112c_{which}.onnx"
        if not onnx.exists():
            m = Trunc(full, which).eval()
            torch.onnx.export(
                m,
                dummy,
                str(onnx),
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
            print("wrote", onnx, onnx.stat().st_size)
        paths[which] = onnx
    return paths


def build_and_bench(paths: dict[str, Path], host: str) -> list[dict]:
    run(f"mkdir -p {REMOTE}", host, timeout=30)
    put(str(ROOT / "bench/trt_bench.py"), "/path/to/board/project", host)
    rows = []
    for which, onnx in paths.items():
        eng = f"{REMOTE}/{onnx.stem}_fp16.plan"
        put(str(onnx), f"{REMOTE}/{onnx.name}", host)
        code, out, _ = run(f"test -s {eng} && echo OK || echo MISS", host)
        if "MISS" in (out or ""):
            print(f"build {which}...", flush=True)
            cmd = (
                f"{TRTEXEC} --onnx={REMOTE}/{onnx.name} --fp16 --saveEngine={eng} "
                f"--memPoolSize=workspace:2048M"
            )
            code, o, e = run(cmd, host, timeout=1800)
            if code != 0:
                raise RuntimeError((o or "") + (e or ""))
        # lock clocks once
        run("sudo nvpmodel -m 2; sudo jetson_clocks", host, timeout=60)
        cmd = (
            f"python3 {BENCH} --engine {eng} --device orin --backend trt_fp16 "
            f"--h 112 --w 112 --warmup 50 --iters 1000"
        )
        code, o, e = run(cmd, host, timeout=300)
        text = (o or "") + (e or "")
        if code != 0:
            raise RuntimeError(text[-1500:])
        # parse p50
        import re

        m = re.search(r"latency_us_p50=([0-9.]+)", text)
        p50_us = float(m.group(1)) if m else float("nan")
        row = {"exit": which, "p50_ms": round(p50_us / 1000.0, 4), "engine": eng}
        print(f"OK {which} p50={row['p50_ms']} ms", flush=True)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="BOARD_IP_REDACTED")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    paths = export_truncated()
    if args.export_only:
        return 0
    rows = build_and_bench(paths, args.host)
    # derive seg from cum (same protocol as Linux boards)
    order = ["e2", "e3", "e4", "final"]
    by = {r["exit"]: r["p50_ms"] for r in rows}
    cum = [by[k] for k in order]
    seg = [0.0, cum[0], cum[1] - cum[0], cum[2] - cum[1], cum[3] - cum[2]]
    feed = {
        "device": "orin",
        "dtype": "fp16",
        "seg_ms": seg,
        "cum_exit_ms": cum,
        "full_ms": cum[-1],
        "source": "results/orin_v4/exit_segment_latency (TRT truncated engines)",
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "summary.json").write_text(
        json.dumps({"rows": rows, "controller_feed": feed}, indent=2),
        encoding="utf-8",
    )
    with (RESULT / "segment_latency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exit", "p50_ms", "engine"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(feed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
