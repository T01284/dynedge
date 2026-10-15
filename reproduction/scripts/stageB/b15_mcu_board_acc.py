#!/usr/bin/env python3
"""B15: full-test accuracy of the ESP-DL G3 anchors on the board (no power meter needed).

Per model: embed the deployed .espdl, build/flash the *_eval firmware, repeat the 8-image
logits gate (embedded images), check that streamed images reproduce all host_ref INT8
logits, then run the whole test set through tools/espdl_board_eval.py under the deployed
(pil) and the training (train) preprocessing.

  python scripts/stageB/b15_mcu_board_acc.py --target esp32s3 --port COM31 [--only ferplus]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from espdl_board_eval import OUT, Board, gray_pil  # noqa: E402
from espdl_pack_ref_images import local_path  # noqa: E402

MODELS = ROOT / "deploy/mcu_espdl/models"
FW = ROOT / "deploy/mcu_espdl/firmware"
ANCHORS = [
    ("g3_w035_d2222_s64g_ferplus", "ferplus"),
    ("g3_w035_d2233_s64g_ferplus", "ferplus"),
    ("g3_w035_d2334_s64g_ferplus", "ferplus"),
    ("g3_w05_d3333_s64g_ferplus", "ferplus"),
    ("g3_w035_d2222_s64g_expw", "expw"),
    ("g3_w035_d2233_s64g_expw", "expw"),
    ("g3_w035_d2344_s64g_expw", "expw"),
    ("g3_w05_d3333_s64g_expw", "expw"),
]


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=str(cwd)).returncode != 0:
        raise SystemExit(f"failed: {cmd}")


def stream_check(port: str, ref_path: Path) -> float:
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    b = Board(port)
    b.ask("CMD NORM rgb", "NORM")
    worst = 0.0
    for i, s in enumerate(ref["samples"]):
        _, exp, raw = b.infer(i, gray_pil(local_path(s["path"])))
        worst = max(worst, float(np.max(np.abs(np.array(raw) * 2.0**exp - np.array(s["int8_logits"])))))
    b.ser.close()
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="esp32s3")
    ap.add_argument("--port", default="COM31")
    ap.add_argument("--only", default="")
    ap.add_argument("--prep", default="pil,train")
    ap.add_argument("--fw", type=Path, default=FW,
                    help="firmware tree; use a separate copy per board to run boards in parallel")
    args = ap.parse_args()
    env = f"{args.target}_eval"
    fw = args.fw.resolve()
    out_dir = OUT / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, dataset in ANCHORS:
        if args.only and args.only not in name:
            continue
        if all((out_dir / f"{name}_{p}.json").exists() for p in args.prep.split(",")):
            print(f"SKIP {name}", flush=True)
            continue
        print(f"\n===== {name} ({args.target}) =====", flush=True)
        espdl = MODELS / f"{name}_{args.target}.espdl"
        ref = MODELS / f"{name}_{args.target}_host_ref.json"
        run([sys.executable, "tools/espdl_pack_ref_images.py", "--ref", str(ref), "--n", "8", "--fw", str(fw)])
        run([sys.executable, "tools/espdl_embed_model.py", "--espdl", str(espdl),
             "--out", str(fw / "main/model_espdl_data.c")])
        run([str(fw / "pio.cmd"), "run", "-e", env, "-t", "upload", "--upload-port", args.port], cwd=fw)
        gate = out_dir / f"{name}_gate8.json"
        run([sys.executable, "tools/espdl_board_check.py", "--port", args.port, "--ref", str(ref), "--out", str(gate)])
        d = stream_check(args.port, ref)
        print(f"GATE {name}: streamed host_ref max|dlogit|={d}", flush=True)
        g = json.loads(gate.read_text(encoding="utf-8"))
        g["stream_max_abs_dlogit"] = d
        gate.write_text(json.dumps(g, indent=2), encoding="utf-8")
        cmd = [sys.executable, "tools/espdl_board_eval.py", "--port", args.port, "--model", name,
               "--target", args.target, "--dataset", dataset, "--prep", args.prep]
        for attempt in range(5):
            print("+", " ".join(cmd), flush=True)
            if subprocess.run(cmd, cwd=str(ROOT)).returncode == 0:
                break
            print(f"RETRY {name} attempt {attempt + 1}: board link lost, resuming in 30 s", flush=True)
            time.sleep(30)
        else:
            raise SystemExit(f"failed: {cmd}")
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
