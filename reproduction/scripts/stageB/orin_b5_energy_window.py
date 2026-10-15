#!/usr/bin/env python3
"""B5: Orin TensorRT energy per inference from a window regression (P5S1, idle-subtracted).

The published Orin energies wrap a whole `python3 trt_bench.py` call (imports, CUDA context,
engine load) around N=1000 inferences, so start-up dominates. Here every (anchor, precision)
is run at N in {1000, 5000, 20000}, --reps times each, in a shuffled order (seed 7), each
window with its own idle segment; E_win = e*N + c is fitted per (anchor, precision), and the
slope e is the start-up-free net energy per inference. The bench P50 of every window is
recorded because each TensorRT process settles on one of two latency levels.
Same session protocol as deploy/orin/scripts/run_p5s1_energy_matrix.py (MAXN_SUPER +
jetson_clocks, warmup process outside the window, host-side marks).

Usage:
  python scripts/stageB/orin_b5_energy_window.py --port COM40
  python scripts/stageB/orin_b5_energy_window.py --port COM40 --reps 1 --ns 1000 --anchors <stem>  # smoke
  python scripts/stageB/orin_b5_energy_window.py --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "orin_energy", ROOT / "deploy/orin/scripts/run_p5s1_energy_matrix.py"
)
E = importlib.util.module_from_spec(_spec)
sys.modules["orin_energy"] = E
assert _spec.loader is not None
_spec.loader.exec_module(E)

OUT = ROOT / "results/stageB/b5_energy_window/orin"
DEFAULT_ANCHORS = [E.ANCHORS[3], E.ANCHORS[0], E.ANCHORS[5]]  # Table II mid, smallest, largest
PRECS = ("int8", "fp16")
RUN_FIELDS = [
    "tag", "precision", "n_infer", "rep", "active_s", "n_active_samples", "idle_s",
    "p_idle_w", "p_idle_sample_std_w", "p_active_w", "energy_mj_per_infer_net",
    "energy_mj_per_infer_gross", "bench_p50_us", "bench_mean_us", "csv",
]


def one(stem: str, prec: str, n_infer: int, rep: int, args) -> dict:
    tag = E.short_tag(stem)
    hw = E.hw_of(stem)
    label = f"{tag}_{prec}_N{n_infer}_r{rep}"
    out_csv = OUT / "raw" / f"{label}.csv"
    sess = E.CaptureSession(args.port)
    sess.start()
    log = ""
    try:
        time.sleep(1.0)
        sess.mark("idle_start")
        time.sleep(args.idle_s)
        sess.mark("idle_stop")
        E.run_trt(stem, prec, hw, args.warmup, 1, timeout=180)
        sess.mark("start")
        log = E.run_trt(stem, prec, hw, 0, n_infer, timeout=max(300, int(n_infer * 0.01) + 120))
        sess.mark("stop")
        time.sleep(0.3)
    finally:
        rows = sess.stop()
    meta = {**sess.meta, "device": "orin", "label": label, "stem": stem, "precision": prec,
            "n_infer": n_infer, "idle_s": args.idle_s, "warmup_process": args.warmup,
            "port": args.port, "n_samples": len(rows), "board_log_tail": log[-800:]}
    E.write_csv(out_csv, rows, meta)
    s = E.summarize_file(out_csv, n_infer, label)

    t0 = [float(r["host_s"]) for r in rows if r["mark"] == "idle_start"][0]
    t1 = [float(r["host_s"]) for r in rows if r["mark"] == "idle_stop"][-1]
    a0 = [float(r["host_s"]) for r in rows if r["mark"] == "start"][0]
    a1 = [float(r["host_s"]) for r in rows if r["mark"] == "stop"][-1]
    idle_p = [float(r["power_w"]) for r in rows if t0 <= float(r["host_s"]) <= t1]
    n_act = sum(1 for r in rows if a0 <= float(r["host_s"]) <= a1)
    p50 = re.search(r"latency_us_p50=([\d.]+)", log)
    mean = re.search(r"latency_us_mean=([\d.]+)", log)
    rec = {
        "tag": tag, "precision": prec, "n_infer": n_infer, "rep": rep,
        "active_s": round(s["active_s"], 4), "n_active_samples": n_act,
        "idle_s": round(s["idle_s"], 4), "p_idle_w": round(s["p_idle_w"], 6),
        "p_idle_sample_std_w": round(statistics.pstdev(idle_p), 6),
        "p_active_w": round(s["p_active_w"], 6),
        "energy_mj_per_infer_net": round(s["energy_mj_per_infer_net"], 6),
        "energy_mj_per_infer_gross": round(s["energy_mj_per_infer_gross"], 6),
        "bench_p50_us": float(p50.group(1)) if p50 else "",
        "bench_mean_us": float(mean.group(1)) if mean else "",
        "csv": str(out_csv.relative_to(ROOT)).replace("\\", "/"),
    }
    print(f"OK {label} act={rec['active_s']}s net={rec['energy_mj_per_infer_net']:.4f} "
          f"Pidle={rec['p_idle_w']:.3f} p50={rec['bench_p50_us']}", flush=True)
    return rec


def fit(rs: list[dict]) -> dict:
    n = np.array([float(r["n_infer"]) for r in rs])
    e_win = np.array([float(r["energy_mj_per_infer_net"]) for r in rs]) * n
    r = stats.linregress(n, e_win)
    tcrit = stats.t.ppf(0.975, len(n) - 2)
    return {
        "e_mj_per_infer": round(r.slope, 4),
        "e_ci95": [round(r.slope - tcrit * r.stderr, 4), round(r.slope + tcrit * r.stderr, 4)],
        "c_mj_per_window": round(r.intercept, 1),
        "c_ci95": [round(r.intercept - tcrit * r.intercept_stderr, 1),
                   round(r.intercept + tcrit * r.intercept_stderr, 1)],
        "r2": round(r.rvalue ** 2, 6), "n_windows": int(len(n)),
    }


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for tag, prec in sorted({(r["tag"], r["precision"]) for r in rows}):
        rs = [r for r in rows if r["tag"] == tag and r["precision"] == prec]
        per_n = {}
        for n in sorted({int(r["n_infer"]) for r in rs}):
            g = [r for r in rs if int(r["n_infer"]) == n]
            net = [float(r["energy_mj_per_infer_net"]) for r in g]
            p50 = [float(r["bench_p50_us"]) for r in g if r["bench_p50_us"] != ""]
            per_n[str(n)] = {
                "n_reps": len(g),
                "active_s_mean": round(statistics.mean(float(r["active_s"]) for r in g), 2),
                "net_mean": round(statistics.mean(net), 4),
                "net_sd": round(statistics.stdev(net), 4) if len(net) > 1 else 0.0,
                "p50_us_min": min(p50) if p50 else None, "p50_us_max": max(p50) if p50 else None,
            }
        idle = [float(r["p_idle_w"]) for r in rs]
        out[f"{tag}_{prec}"] = {
            "fit": fit(rs) if len({int(r['n_infer']) for r in rs}) > 1 else None,
            "by_n": per_n,
            "p_idle_w_mean": round(statistics.mean(idle), 4),
            "p_idle_w_sd": round(statistics.stdev(idle), 4) if len(idle) > 1 else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM40")
    ap.add_argument("--ns", default="1000,5000,20000")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS))
    ap.add_argument("--precisions", default=",".join(PRECS))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    runs_csv = OUT / "runs.csv"
    done = list(csv.DictReader(runs_csv.open(encoding="utf-8"))) if runs_csv.exists() else []
    if not args.summarize_only:
        anchors = [a for a in args.anchors.split(",") if a]
        precs = [p for p in args.precisions.split(",") if p]
        E.prep_models(anchors, [f"trt_{p}" for p in precs])
        ns = [int(n) for n in args.ns.split(",")]
        jobs = [(s, p, n, r) for s in anchors for p in precs for n in ns for r in range(1, args.reps + 1)]
        random.Random(7).shuffle(jobs)
        keys = {(d["tag"], d["precision"], int(d["n_infer"]), int(d["rep"])) for d in done}
        todo = [j for j in jobs if (E.short_tag(j[0]), j[1], j[2], j[3]) not in keys]
        print(f"jobs={len(jobs)} todo={len(todo)}", flush=True)
        for i, (stem, prec, n, r) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}]", flush=True)
            try:
                done.append(one(stem, prec, n, r, args))
            except Exception as ex:  # noqa: BLE001
                print(f"FAIL {E.short_tag(stem)} {prec} N{n} r{r}: {ex}", flush=True)
                continue
            with runs_csv.open("w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                wr.writeheader()
                wr.writerows(done)
    summ = {"by_config": summarize(done), "protocol": {
        "meter": "P5S1 100 Hz, Orin DC high side", "backend": "TensorRT trt_bench.py, one process per window",
        "power_mode": "MAXN_SUPER + jetson_clocks", "idle_s": args.idle_s,
        "warmup_process_outside_window": args.warmup, "order": "shuffled, random.Random(7)",
        "fit": "E_win = e*N + c over all windows of a config; e = net mJ/inf",
    }}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for k, v in summ["by_config"].items():
        print(k, json.dumps(v["fit"]), f"idle={v['p_idle_w_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
