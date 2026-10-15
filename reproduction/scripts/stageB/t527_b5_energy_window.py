#!/usr/bin/env python3
"""B5: energy-window sensitivity on the T527 NPU with the P5S1 (idle-subtracted net mJ/inf).

For each anchor, the number of inferences N is chosen so that the active window lasts
about T seconds, T in {2, 5, 10, 30}. Every (anchor, T) is repeated --reps times in a
shuffled order (seed 7), each with its own idle segment, so the spread across reps
contains idle drift as well as integration noise. Same protocol as
deploy/t527/scripts/run_p5s1_energy_matrix.py (performance governor, warmup outside
the window, host-side marks).

Usage:
  python scripts/stageB/t527_b5_energy_window.py --port COM40
  python scripts/stageB/t527_b5_energy_window.py --port COM40 --reps 1 --windows 2   # smoke
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "t527_energy", ROOT / "deploy/t527/scripts/run_p5s1_energy_matrix.py"
)
E = importlib.util.module_from_spec(_spec)
sys.modules["t527_energy"] = E
assert _spec.loader is not None
_spec.loader.exec_module(E)

OUT = ROOT / "results/stageB/b5_energy_window/t527" / ("" if E.NBG_DIR.name == "v4_npu" else E.NBG_DIR.name)
DEFAULT_ANCHORS = [E.ANCHORS[3], E.ANCHORS[0]]  # Table II operating point, shortest inference
RUN_FIELDS = [
    "tag", "window_s", "rep", "n_infer", "active_s", "n_active_samples", "idle_s",
    "p_idle_w", "p_idle_sample_std_w", "p_active_w", "energy_mj_per_infer_net",
    "energy_mj_per_infer_gross", "csv",
]


def per_infer_s(remote: str) -> float:
    """Board-side wall time per vpm_run loop (includes per-loop logging)."""
    n = 300
    _, out, _ = E.run(
        f"cd {remote} && s=$(date +%s%N); vpm_run -s sample.txt -l {n} -b 1 >/dev/null 2>&1; "
        f"e=$(date +%s%N); echo $(( (e-s)/1000 ))",
        E.HOST,
        timeout=120,
    )
    return int(out.strip().splitlines()[-1]) / 1e6 / n


def one(stem: str, remote: str, window_s: float, rep: int, n_infer: int, args) -> dict:
    tag = E.short_tag(stem)
    label = f"{tag}_T{int(window_s)}_r{rep}"
    out_csv = OUT / "raw" / f"{label}.csv"
    sess = E.CaptureSession(args.port)
    sess.start()
    try:
        time.sleep(1.0)
        sess.mark("idle_start")
        time.sleep(args.idle_s)
        sess.mark("idle_stop")
        E.run_npu(remote, args.warmup, timeout=120)
        sess.mark("start")
        E.run_npu(remote, n_infer, timeout=max(180, int(window_s * 3) + 60))
        sess.mark("stop")
        time.sleep(0.3)
    finally:
        rows = sess.stop()
    meta = {**sess.meta, "device": "t527", "label": label, "stem": stem, "n_infer": n_infer,
            "window_target_s": window_s, "idle_s": args.idle_s, "warmup": args.warmup,
            "port": args.port, "n_samples": len(rows)}
    E.write_csv(out_csv, rows, meta)
    s = E.summarize_file(out_csv, n_infer, "t527", label)

    t0 = [float(r["host_s"]) for r in rows if r["mark"] == "idle_start"][0]
    t1 = [float(r["host_s"]) for r in rows if r["mark"] == "idle_stop"][-1]
    a0 = [float(r["host_s"]) for r in rows if r["mark"] == "start"][0]
    a1 = [float(r["host_s"]) for r in rows if r["mark"] == "stop"][-1]
    idle_p = [float(r["power_w"]) for r in rows if t0 <= float(r["host_s"]) <= t1]
    n_act = sum(1 for r in rows if a0 <= float(r["host_s"]) <= a1)
    rec = {
        "tag": tag, "window_s": window_s, "rep": rep, "n_infer": n_infer,
        "active_s": round(s["active_s"], 4), "n_active_samples": n_act,
        "idle_s": round(s["idle_s"], 4), "p_idle_w": round(s["p_idle_w"], 6),
        "p_idle_sample_std_w": round(statistics.pstdev(idle_p), 6),
        "p_active_w": round(s["p_active_w"], 6),
        "energy_mj_per_infer_net": round(s["energy_mj_per_infer_net"], 6),
        "energy_mj_per_infer_gross": round(s["energy_mj_per_infer_gross"], 6),
        "csv": str(out_csv.relative_to(ROOT)).replace("\\", "/"),
    }
    print(f"OK {label} N={n_infer} act={rec['active_s']}s net={rec['energy_mj_per_infer_net']:.4f} "
          f"Pidle={rec['p_idle_w']:.4f}", flush=True)
    return rec


def ci95(x: list[float]) -> tuple[float, float, float, float]:
    a = np.asarray(x, dtype=float)
    m, sd = float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0
    h = float(stats.t.ppf(0.975, a.size - 1) * sd / np.sqrt(a.size)) if a.size > 1 else float("nan")
    return m, sd, m - h, m + h


def summarize(rows: list[dict]) -> dict:
    out: dict = {"by_anchor": {}}
    for tag in sorted({r["tag"] for r in rows}):
        rs = [r for r in rows if r["tag"] == tag]
        per_w = {}
        for w in sorted({float(r["window_s"]) for r in rs}):
            ws = [r for r in rs if float(r["window_s"]) == w]
            net = [float(r["energy_mj_per_infer_net"]) for r in ws]
            m, sd, lo, hi = ci95(net)
            per_w[str(int(w))] = {
                "n_reps": len(ws), "n_infer": int(ws[0]["n_infer"]),
                "active_s_mean": round(statistics.mean(float(r["active_s"]) for r in ws), 3),
                "samples_per_window": round(statistics.mean(int(r["n_active_samples"]) for r in ws), 1),
                "net_mean": round(m, 5), "net_sd": round(sd, 5),
                "net_ci95": [round(lo, 5), round(hi, 5)],
                "cv": round(sd / m, 4) if m else None,
            }
        ref = per_w.get(max(per_w, key=float), {}).get("net_mean")
        for v in per_w.values():
            v["rel_to_longest"] = round(v["net_mean"] / ref - 1, 4) if ref else None
        idle = [float(r["p_idle_w"]) for r in rs]
        out["by_anchor"][tag] = {
            "windows": per_w,
            "idle": {
                "n": len(idle), "mean_w": round(statistics.mean(idle), 5),
                "sd_between_windows_w": round(statistics.stdev(idle), 5) if len(idle) > 1 else 0.0,
                "mean_within_window_sample_sd_w": round(
                    statistics.mean(float(r["p_idle_sample_std_w"]) for r in rs), 5),
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM40")
    ap.add_argument("--windows", default="2,5,10,30")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS))
    ap.add_argument("--all-anchors", action="store_true", help="all six NPU anchors of this NBG set")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    runs_csv = OUT / "runs.csv"
    done = list(csv.DictReader(runs_csv.open(encoding="utf-8"))) if runs_csv.exists() else []
    if not args.summarize_only:
        E.lock_perf()
        windows = [float(w) for w in args.windows.split(",")]
        anchors = list(E.ANCHORS) if args.all_anchors else [a for a in args.anchors.split(",") if a]
        remotes, t_inf = {}, {}
        for stem in anchors:
            remotes[stem] = E.ensure_npu(stem)
            E.run_npu(remotes[stem], args.warmup, timeout=120)
            t_inf[stem] = per_infer_s(remotes[stem])
            print(f"{E.short_tag(stem)}: {t_inf[stem]*1e3:.3f} ms/loop", flush=True)
        jobs = [(s, w, r) for s in anchors for w in windows for r in range(1, args.reps + 1)]
        random.Random(7).shuffle(jobs)
        keys = {(d["tag"], float(d["window_s"]), int(d["rep"])) for d in done}
        for stem, w, r in jobs:
            if (E.short_tag(stem), w, r) in keys:
                continue
            n = max(1, round(w / t_inf[stem]))
            try:
                done.append(one(stem, remotes[stem], w, r, n, args))
            except Exception as ex:  # noqa: BLE001
                print(f"FAIL {E.short_tag(stem)} T{w} r{r}: {ex}", flush=True)
                continue
            with runs_csv.open("w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                wr.writeheader()
                wr.writerows(done)
    summ = summarize(done)
    summ["protocol"] = {
        "meter": "P5S1 100 Hz, high side of the 12 V board input",
        "backend": "T527 VIPLite NBG (vpm_run)", "governor": "performance",
        "idle_s": args.idle_s, "warmup_outside_window": args.warmup,
        "order": "shuffled, random.Random(7)",
        "ci": "Student t over reps",
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for tag, v in summ["by_anchor"].items():
        print(tag, json.dumps(v["idle"]))
        for w, s in v["windows"].items():
            print(f"  T={w:>2}s N={s['n_infer']:>5} samples={s['samples_per_window']:>6} "
                  f"net={s['net_mean']:.4f}±{s['net_sd']:.4f} CI{s['net_ci95']} cv={s['cv']} "
                  f"rel={s['rel_to_longest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
