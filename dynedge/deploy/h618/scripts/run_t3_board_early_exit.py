#!/usr/bin/env python3
"""H618 early-exit board suite — mirror T527 §8.15 + segment calibration.

Same multi-exit bake NCNN as T527 T3 (ferplus w0.5 d3333 @112).
Measures:
  - force_final baseline
  - τ ∈ {0.7, 0.8, 0.85, 0.9}
  - forced exits 0/1/2 (+ force_final) → cum wall clock per exit → seg = Δcum

Formal threads=4; n=80 eval500 hw112 bins.
"""
from __future__ import annotations

import json
import re
import tarfile
import tempfile
from pathlib import Path

from ssh_h618 import put, run

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/h618_v4/t3_early_exit"
T3_NCNN = ROOT / "results/t527_v4/t3_early_exit/ncnn"
BIN = ROOT / "deploy/ncnn_edge/build-linux-aarch64-acc/fer_early_exit"
BIN_DIR = ROOT / "results/t527_v4/eval500_bins/hw112"
THREADS = 4
N = 80
TAUS = (0.7, 0.8, 0.85, 0.9)
# LUT reference from G5 controller summary (to be calibrated by this run)
LUT_CUM = [9.607, 11.095, 12.379, 12.520]


def parse_result(text: str) -> dict:
    m = re.search(r"^RESULT .+$", text, re.M)
    if not m:
        return {"raw": text}
    row = {"raw": m.group(0)}
    for p in m.group(0).split()[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        row[k] = v
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (T3_NCNN / "m.ncnn.param").exists():
        raise SystemExit(f"missing multi-exit ncnn under {T3_NCNN}")
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}")

    bins = sorted(BIN_DIR.glob("*.bin"))[:N]
    # Forced constant-exit lists for segment cum measurement
    for k in range(3):
        (OUT / f"exits_force_e{k}.txt").write_text(
            "\n".join([str(k)] * N) + "\n", encoding="utf-8"
        )

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "bins.tar"
        with tarfile.open(tar_path, "w") as tar:
            for p in bins:
                tar.add(p, arcname=p.name)
        run("rm -rf /root/h618_t3_ee; mkdir -p /root/h618_t3_ee/bins /root/h618_t3_ee/models")
        put(str(BIN), "/root/h618_t3_ee/fer_early_exit")
        put(str(T3_NCNN / "m.ncnn.param"), "/root/h618_t3_ee/models/m.ncnn.param")
        put(str(T3_NCNN / "m.ncnn.bin"), "/root/h618_t3_ee/models/m.ncnn.bin")
        put(str(tar_path), "/root/h618_t3_ee/bins.tar")
        for k in range(3):
            put(str(OUT / f"exits_force_e{k}.txt"), f"/root/h618_t3_ee/exits_force_e{k}.txt")

    code, out, err = run(
        "cd /root/h618_t3_ee && tar xf bins.tar -C bins && "
        "for f in bins/*; do echo /root/h618_t3_ee/$f; done > list80.txt && "
        "chmod +x fer_early_exit && wc -l list80.txt",
        timeout=120,
    )
    print(out, err, "prep", code)
    if code != 0:
        return code

    common = (
        "./fer_early_exit --param models/m.ncnn.param --bin models/m.ncnn.bin "
        f"--list list80.txt --h 112 --w 112 --c 3 --threads {THREADS} "
        "--out0 out0 --out1 out1 --out2 out2 --out3 out3"
    )

    results: dict = {}

    # --- force_final ---
    code, o, e = run(f"cd /root/h618_t3_ee && {common} --force-final --tau 0.85", timeout=300)
    print("force_final", o)
    (OUT / "board_force_final.txt").write_text((o or "") + "\n" + (e or ""), encoding="utf-8")
    results["force_final"] = parse_result(o or "")
    if code != 0:
        return code

    # --- τ sweep ---
    for tau in TAUS:
        code, o, e = run(
            f"cd /root/h618_t3_ee && {common} --tau {tau}", timeout=300
        )
        key = f"tau_{tau:g}"
        print(key, o)
        (OUT / f"board_{key}.txt").write_text((o or "") + "\n" + (e or ""), encoding="utf-8")
        results[key] = parse_result(o or "")
        if code != 0:
            return code

    # --- forced cum @ each early exit ---
    cum_board = []
    for k in range(3):
        code, o, e = run(
            f"cd /root/h618_t3_ee && {common} --exits exits_force_e{k}.txt --tau 0.85",
            timeout=300,
        )
        key = f"force_exit_{k}"
        print(key, o)
        (OUT / f"board_{key}.txt").write_text((o or "") + "\n" + (e or ""), encoding="utf-8")
        results[key] = parse_result(o or "")
        if code != 0:
            return code
        cum_board.append(float(results[key].get("p50_ms", "nan")))
    cum_board.append(float(results["force_final"].get("p50_ms", "nan")))

    seg_board = [cum_board[0]] + [cum_board[i] - cum_board[i - 1] for i in range(1, 4)]
    lut_seg = [LUT_CUM[0]] + [LUT_CUM[i] - LUT_CUM[i - 1] for i in range(1, 4)]

    def p50(row: dict) -> float:
        return float(row.get("p50_ms", "nan"))

    ff = p50(results["force_final"])
    speedups = {}
    for tau in TAUS:
        key = f"tau_{tau:g}"
        t = p50(results[key])
        speedups[key] = round(ff / t, 4) if t > 0 else None

    summary = {
        "status": "ok_h618_t3_suite",
        "note": (
            "Mirror T527 §8.15: same multi-exit bake; τ sweep + force_final; "
            "forced e0..e2 + FF give board cum/seg for C3 budget-band calibration"
        ),
        "device": "h618",
        "threads": THREADS,
        "n": N,
        "model": "results/t527_v4/t3_early_exit/ncnn (shared multi-exit)",
        "board": results,
        "speedup_p50_vs_force_final": speedups,
        "segment_calibration": {
            "lut_cum_exit_ms": LUT_CUM,
            "lut_seg_ms_approx": [round(x, 3) for x in lut_seg],
            "board_cum_p50_ms": [round(x, 3) for x in cum_board],
            "board_seg_p50_ms": [round(x, 3) for x in seg_board],
            "board_over_lut_cum": [
                round(cum_board[i] / LUT_CUM[i], 3) if LUT_CUM[i] > 0 else None
                for i in range(4)
            ],
            "exit_labels": ["exit2", "exit3", "exit4", "final"],
        },
    }
    (OUT / "t3_board_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
