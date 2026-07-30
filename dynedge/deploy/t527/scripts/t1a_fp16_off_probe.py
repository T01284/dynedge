#!/usr/bin/env python3
"""T1a: board NCNN FP32 (fp16=off) vs host ONNX on 20 images."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_t527 import put, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
STEM = "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64"
MODEL = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
ONNX = ROOT / "results" / "h618_onnx_v4" / f"{STEM}.onnx"
ACC = ROOT / "deploy" / "ncnn_edge" / "build-linux-aarch64-acc" / "fer_ncnn_acc"
BINS = ROOT / "results" / "t527_v4" / "eval500_bins" / "hw64"
LIST = ROOT / "results" / "t527_v4" / "eval500_bins" / "list_hw64.txt"
REMOTE = "/root/fer-dynedge"
N = 20


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    names = [ln.strip().split("/")[-1] for ln in LIST.read_text(encoding="utf-8").splitlines() if ln.strip()][
        :N
    ]
    # short list on board
    run(f"mkdir -p {REMOTE}/acc500/bins/hw64 {REMOTE}/bin {REMOTE}/models/v4_predistill")
    put(str(ACC), f"{REMOTE}/bin/fer_ncnn_acc")
    run(f"chmod +x {REMOTE}/bin/fer_ncnn_acc")
    put(str(MODEL / f"{STEM}.param"), f"{REMOTE}/models/v4_predistill/{STEM}.param")
    put(str(MODEL / f"{STEM}.bin"), f"{REMOTE}/models/v4_predistill/{STEM}.bin")

    short = ROOT / "results" / "t527_v4" / "t1a_list20.txt"
    lines = []
    for name in names:
        local = BINS / name
        remote = f"{REMOTE}/acc500/bins/hw64/{name}"
        put(str(local), remote)
        lines.append(remote)
    short.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    put(str(short), f"{REMOTE}/acc500/t1a_list20.txt")

    cmd = (
        f"{REMOTE}/bin/fer_ncnn_acc --fp32-only "
        f"--fp32-param {REMOTE}/models/v4_predistill/{STEM}.param "
        f"--fp32-bin {REMOTE}/models/v4_predistill/{STEM}.bin "
        f"--list {REMOTE}/acc500/t1a_list20.txt --h 64 --w 64 --c 3 --threads 2 --limit {N}"
    )
    code, out, err = run(cmd, timeout=300)
    text = out + err
    print(text)
    if code != 0 and "RESULT" not in text:
        print("T1a FAIL: board exit", code)
        return 1

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    rows = []
    for m in re.finditer(r"^LOGITS idx=(\d+) path=(\S+) finite=(\d) n=(\d+) vals=(.*)$", text, re.M):
        idx = int(m.group(1))
        finite = int(m.group(3))
        vals = np.asarray([float(x) for x in m.group(5).split(",") if x], dtype=np.float32)
        name = Path(m.group(2)).name
        chw = np.fromfile(BINS / name, dtype=np.float32).reshape(3, 64, 64)
        z_onnx = np.asarray(sess.run([out_name], {inp: chw[None]})[0], dtype=np.float32).reshape(-1)
        cos = cosine(vals, z_onnx) if finite and np.isfinite(z_onnx).all() else float("nan")
        rows.append({"idx": idx, "finite": finite, "cos": cos, "match": int(vals.argmax() == z_onnx.argmax())})
        print(f"  [{idx}] finite={finite} cos={cos:.6f} match={rows[-1]['match']}")

    n = len(rows)
    n_fin = sum(r["finite"] for r in rows)
    cos_ok = [r["cos"] for r in rows if r["finite"] and np.isfinite(r["cos"])]
    mean_cos = float(np.mean(cos_ok)) if cos_ok else float("nan")
    print(f"SUMMARY n={n} finite={n_fin}/{n} mean_cos={mean_cos:.6f}")
    ok = n_fin == n and n >= N and mean_cos > 0.999
    print("T1a", "PASS" if ok else "FAIL", "(root-cause FP16 confirmed)" if ok else "(need T1b)")
    out_json = ROOT / "results" / "t527_v4" / "t1a_fp16_off_probe.json"
    import json

    out_json.write_text(
        json.dumps(
            {
                "stem": STEM,
                "n": n,
                "finite": n_fin,
                "mean_cos_vs_onnx": mean_cos,
                "pass": ok,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
