#!/usr/bin/env python3
"""Smoke: build Orin TRT FP16 for T3 multi-exit bake ONNX (segment-latency precursor)."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SCRIPTS))

from ssh_orin import put, run  # noqa: E402

ONNX = _ROOT / "results/t527_v4/t3_early_exit/ncnn/m.pnnx.onnx"
REMOTE = "/path/to/board/project"
REMOTE_ONNX = f"{REMOTE}/models/multi_exit/m.pnnx.onnx"
REMOTE_ENG = f"{REMOTE}/models/multi_exit/m_fp16.plan"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def main() -> int:
    if not ONNX.exists():
        raise SystemExit(f"missing {ONNX}")
    run(f"mkdir -p {REMOTE}/models/multi_exit", timeout=30)
    put(str(ONNX), REMOTE_ONNX)
    code, out, _ = run(f"test -s {REMOTE_ENG} && echo OK || echo MISS")
    if "OK" in (out or ""):
        print("engine exists", flush=True)
        return 0
    cmd = (
        f"{TRTEXEC} --onnx={REMOTE_ONNX} --fp16 --saveEngine={REMOTE_ENG} "
        f"--memPoolSize=workspace:2048M"
    )
    print("building multi-exit FP16...", flush=True)
    code, out, err = run(cmd, timeout=1800)
    text = (out or "") + (err or "")
    if code != 0:
        print(text[-3000:])
        return code
    print("built", REMOTE_ENG, flush=True)
    # quick shape probe via trtexec load
    code, out, err = run(
        f"{TRTEXEC} --loadEngine={REMOTE_ENG} --warmUp=200 --duration=3",
        timeout=120,
    )
    print((out or "")[-1500:])
    print((err or "")[-500:])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
