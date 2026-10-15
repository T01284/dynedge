#!/usr/bin/env python3
"""B5: ESP32-S3 energy-window sensitivity, metered by a Nordic PPK2.

PPK2 powers the board (source 5.0 V) and is owned by `bench.ppk2.hold`. This script
commands the USB-JTAG console (firmware env esp32s3_eval): a 5 s idle, a warmup
`CMD RUN` outside the window, then one tight on-board loop `CMD RUN N` inside it.
Windows T in {2, 5, 10, 30} s, --reps each, shuffled (seed 7). The flashed model is
the Table II anchor g3 w0.35_d2222@64g FER+.

  python -m bench.ppk2.hold --port COM46 --mv 5000 --out results/power/esp32s3_ppk2/stream
  python scripts/stageB/s3_b5_energy_window.py --port COM31
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageB"))
_spec = importlib.util.spec_from_file_location(
    "h618_b5", ROOT / "scripts/stageB/h618_b5_energy_window.py"
)
H618 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(H618)
from t527_b5_energy_window import summarize  # noqa: E402

STREAM = ROOT / "results/power/esp32s3_ppk2/stream"
OUT = ROOT / "results/stageB/b5_energy_window/esp32s3_ppk2"
TAG = "w035_d2222_s64g"
RUN_FIELDS = [
    "tag", "window_s", "rep", "n_infer", "active_s", "n_active_samples", "idle_s",
    "p_idle_w", "p_idle_sample_std_w", "p_active_w", "energy_mj_per_infer_net",
    "energy_mj_per_infer_gross", "source_mv", "run_us_mean",
]
RUN_RE = re.compile(r"kind=run n=(\d+) wall_us=(\d+) run_us_mean=([\d.]+)")


class Board:
    def __init__(self, port: str) -> None:
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = 115200
        self.ser.timeout = 0.5
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def run(self, n: int, timeout: float) -> tuple[float, float]:
        self.ser.write(f"CMD RUN {n}\n".encode())
        self.ser.flush()
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.ser.readline().decode("utf-8", "replace").strip()
            if line.startswith("ERR"):
                raise RuntimeError(line)
            m = RUN_RE.search(line)
            if m:
                return int(m.group(2)) / 1e6 / int(m.group(1)), float(m.group(3))
        raise TimeoutError(f"no RESULT for CMD RUN {n}")


def one(stream, board: Board, window_s: float, rep: int, n_infer: int, args) -> dict:
    H618.stream_alive()
    time.sleep(1.0)
    i0 = time.time()
    time.sleep(args.idle_s)
    i1 = time.time()
    board.run(args.warmup, timeout=max(30.0, args.warmup * 0.2))
    a0 = time.time()
    _, run_us = board.run(n_infer, timeout=max(60.0, n_infer * 0.25))
    a1 = time.time()
    time.sleep(1.8)
    stream.refresh(keep_after=i0 - 1)
    v = H618.source_mv() / 1e3
    ti, ii = stream.window(i0, i1)
    ta, ia = stream.window(a0, a1)
    if len(ti) < 0.9 * args.idle_s * 1000 or len(ta) < 0.9 * (a1 - a0) * 1000:
        raise RuntimeError(f"stream gap: idle {len(ti)} / active {len(ta)} samples")
    import numpy as np
    p_idle = v * float(np.trapezoid(ii, ti)) / (ti[-1] - ti[0])
    e_act = v * float(np.trapezoid(ia, ta))
    t_act = float(ta[-1] - ta[0])
    bins = ii[: len(ii) // 10 * 10].reshape(-1, 10).mean(1) * v
    rec = {
        "tag": args.tag, "window_s": window_s, "rep": rep, "n_infer": n_infer,
        "active_s": round(t_act, 4), "n_active_samples": int(len(ta)),
        "idle_s": round(float(ti[-1] - ti[0]), 4), "p_idle_w": round(p_idle, 6),
        "p_idle_sample_std_w": round(float(bins.std()), 6),
        "p_active_w": round(e_act / t_act, 6),
        "energy_mj_per_infer_net": round((e_act - p_idle * t_act) * 1e3 / n_infer, 6),
        "energy_mj_per_infer_gross": round(e_act * 1e3 / n_infer, 6),
        "source_mv": int(round(v * 1e3)), "run_us_mean": round(run_us, 1),
    }
    print(f"OK {args.tag}_T{int(window_s)}_r{rep} N={n_infer} act={rec['active_s']}s "
          f"net={rec['energy_mj_per_infer_net']:.4f} Pidle={rec['p_idle_w']:.4f} "
          f"Pact={rec['p_active_w']:.4f} run={run_us/1e3:.2f}ms", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM31")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--stream", type=Path, default=STREAM)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", default="g3_w035_d2222_s64g_ferplus (Table II anchor)")
    ap.add_argument("--windows", default="2,5,10,30")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    runs_csv = out / "runs.csv"
    done = list(csv.DictReader(runs_csv.open(encoding="utf-8"))) if runs_csv.exists() else []
    if not args.summarize_only:
        # point the H618 helpers at this session's stream
        H618.STREAM = args.stream
        H618.stream_alive()
        board = Board(args.port)
        info = board.ser  # keep open
        del info
        wall_s, _ = board.run(30, timeout=20)
        print(f"{args.tag}: {wall_s*1e3:.2f} ms/inf (preprocess+run)", flush=True)
        windows = [float(w) for w in args.windows.split(",")]
        jobs = [(w, r) for w in windows for r in range(1, args.reps + 1)]
        random.Random(7).shuffle(jobs)
        keys = {(float(d["window_s"]), int(d["rep"])) for d in done}
        stream = H618.Stream(args.stream)
        stream.refresh(keep_after=time.time())
        todo = [(w, r) for w, r in jobs if (w, r) not in keys]
        print(f"jobs={len(jobs)} todo={len(todo)}", flush=True)
        for i, (w, r) in enumerate(todo, 1):
            n = max(1, round(w / wall_s))
            print(f"[{i}/{len(todo)}]", flush=True)
            try:
                done.append(one(stream, board, w, r, n, args))
            except Exception as ex:  # noqa: BLE001
                print(f"FAIL T{w} r{r}: {ex}", flush=True)
                continue
            with runs_csv.open("w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                wr.writeheader()
                wr.writerows(done)
        board.ser.close()
    summ = summarize(done)
    summ["protocol"] = {
        "meter": "Nordic PPK2 source-meter 5.0 V, 100 kHz raw, 1 kHz block means",
        "board": "USB-JTAG console, CMD RUN tight loop",
        "model": args.model,
        "idle_s": args.idle_s, "warmup_outside_window": args.warmup,
        "order": "shuffled, random.Random(7)", "ci": "Student t over reps",
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for tag, v in summ.get("by_anchor", {}).items():
        print(tag, json.dumps(v["idle"]))
        for w, s in v["windows"].items():
            print(f"  T={w:>2}s N={s['n_infer']:>5} net={s['net_mean']:.4f}+-{s['net_sd']:.4f} "
                  f"CI{s['net_ci95']} cv={s['cv']} rel={s['rel_to_longest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
