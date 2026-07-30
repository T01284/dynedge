#!/usr/bin/env python3
"""Push v4 predistill ONNX anchors to Orin and build TensorRT engines via trtexec.

Usage:
  python deploy/orin/scripts/build_engines.py --precision fp16
  python deploy/orin/scripts/build_engines.py --precision fp16,int8 --limit 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[3]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ssh_orin import DEFAULT_HOST, put, run  # noqa: E402

ONNX_DIR = _ROOT / "results/h618_onnx_v4"
REMOTE = "/path/to/board/project"
REMOTE_ONNX = f"{REMOTE}/models/v4_predistill_onnx"
REMOTE_ENG = f"{REMOTE}/models/v4_predistill_trt"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

ANCHORS = [
    "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64",
    "subnet_ferplus_w0.35_d3333_s64g_seed42_v4_predistill_img64",
    "subnet_ferplus_w0.5_d2222_s96c_seed42_v4_predistill_img96",
    "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_img112",
    "subnet_ferplus_w0.75_d3333_s112c_seed42_v4_predistill_img112",
    "subnet_ferplus_w1_d4444_s112c_seed42_v4_predistill_img112",
]


def lock_clocks(host: str) -> None:
    code, out, err = run(
        "sudo nvpmodel -m 2; sudo jetson_clocks; sudo nvpmodel -q; "
        "sudo jetson_clocks --show 2>&1 | head -20",
        host,
        timeout=60,
    )
    print(out or err, flush=True)
    if code != 0:
        print(f"warn: lock_clocks rc={code}", flush=True)


def ensure_remote_tree(host: str) -> None:
    run(f"mkdir -p {REMOTE_ONNX} {REMOTE_ENG} {REMOTE}/bench {REMOTE}/results", host)


def push_bench(host: str) -> None:
    local = _ROOT / "bench/trt_bench.py"
    put(str(local), f"{REMOTE}/bench/trt_bench.py", host)
    # schema optional import not required at runtime


def push_onnx(stem: str, host: str) -> str:
    local = ONNX_DIR / f"{stem}.onnx"
    if not local.exists():
        raise FileNotFoundError(local)
    remote = f"{REMOTE_ONNX}/{stem}.onnx"
    code, out, _ = run(f"stat -c %s {remote} 2>/dev/null || echo 0", host)
    remote_sz = int((out or "0").strip() or "0")
    if remote_sz != local.stat().st_size:
        print(f"  push {local.name} ({local.stat().st_size} B)", flush=True)
        put(str(local), remote, host)
    else:
        print(f"  skip push {local.name} (size match)", flush=True)
    return remote


def build_one(stem: str, precision: str, host: str, workspace_mib: int) -> str:
    onnx = push_onnx(stem, host)
    eng = f"{REMOTE_ENG}/{stem}_{precision}.plan"
    code, out, _ = run(f"test -s {eng} && echo OK || echo MISS", host)
    if "OK" in (out or ""):
        print(f"  engine exists {Path(eng).name}", flush=True)
        return eng
    flags = ["--onnx=" + onnx, f"--saveEngine={eng}", f"--memPoolSize=workspace:{workspace_mib}M"]
    if precision == "fp16":
        flags.append("--fp16")
    elif precision == "fp32":
        pass
    elif precision == "int8":
        # latency/energy first; accuracy needs real calibrator (later)
        flags.extend(["--int8", "--fp16"])
    else:
        raise ValueError(precision)
    cmd = f"{TRTEXEC} " + " ".join(flags)
    print(f"  build {precision}: {stem}", flush=True)
    code, out, err = run(cmd, host, timeout=1800)
    text = (out or "") + (err or "")
    if code != 0:
        raise RuntimeError(f"trtexec failed rc={code}\n{text[-2500:]}")
    print(f"  built {Path(eng).name}", flush=True)
    return eng


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--precision", default="fp16", help="comma list: fp16,int8,fp32")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workspace-mib", type=int, default=2048)
    ap.add_argument("--skip-lock", action="store_true")
    args = ap.parse_args()

    precisions = [p.strip() for p in args.precision.split(",") if p.strip()]
    anchors = ANCHORS[: args.limit] if args.limit else ANCHORS

    if not args.skip_lock:
        lock_clocks(args.host)
    ensure_remote_tree(args.host)
    push_bench(args.host)

    for stem in anchors:
        print(f"\n== {stem} ==", flush=True)
        for p in precisions:
            build_one(stem, p, args.host, args.workspace_mib)
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
