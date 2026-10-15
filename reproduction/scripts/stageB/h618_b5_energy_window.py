#!/usr/bin/env python3
"""B5: energy-window sensitivity on the H618 (NCNN INT8), metered by a Nordic PPK2.

The PPK2 powers the board in source-meter mode and is owned by `bench.ppk2.hold`,
which streams 1 kHz current means to results/power/h618_ppk2/stream/. This script
marks idle and active windows by host wall-clock time and integrates the stream:
E = V_src * integral(I dt), net = (E_act - P_idle * T_act) / N.

Protocol otherwise matches deploy/h618/scripts/run_p5s1_energy_matrix.py
(performance governor, fer_ncnn_bench 4 threads, warmup outside the window,
N timed inferences with --warmup 0 inside it). Windows T in {2,5,10,30} s,
--reps each, shuffled order (seed 7), own 5 s idle per window.

Usage:
  python -m bench.ppk2.hold --port COM46 --mv 5000          # running in background
  python scripts/stageB/h618_b5_energy_window.py
  python scripts/stageB/h618_b5_energy_window.py --reps 1 --windows 2   # smoke
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "h618_energy", ROOT / "deploy/h618/scripts/run_p5s1_energy_matrix.py"
)
H = importlib.util.module_from_spec(_spec)
sys.modules["h618_energy"] = H
assert _spec.loader is not None
_spec.loader.exec_module(H)
sys.path.insert(0, str(ROOT / "scripts/stageB"))
from t527_b5_energy_window import ci95, summarize  # noqa: E402

STREAM = ROOT / "results/power/h618_ppk2/stream"
OUT = ROOT / "results/stageB/b5_energy_window/h618_ppk2"
DEFAULT_ANCHORS = [
    "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_img112",  # Table II operating point
    "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64",  # shortest inference
]
RUN_FIELDS = [
    "tag", "window_s", "rep", "n_infer", "active_s", "n_active_samples", "idle_s",
    "p_idle_w", "p_idle_sample_std_w", "p_active_w", "energy_mj_per_infer_net",
    "energy_mj_per_infer_gross", "source_mv",
]


class Stream:
    """Incremental reader over the hold.py hourly stream files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pos: dict[str, int] = {}
        self.t = np.empty(0)
        self.i = np.empty(0)

    def refresh(self, keep_after: float) -> None:
        ts, cs = [self.t[self.t >= keep_after]], [self.i[self.t >= keep_after]]
        for f in sorted(self.root.glob("stream_*.csv")):
            with f.open("rb") as fh:
                fh.seek(self.pos.get(f.name, 0))
                chunk = fh.read()
            cut = chunk.rfind(b"\n") + 1
            self.pos[f.name] = self.pos.get(f.name, 0) + cut
            rows = [ln.split(b",") for ln in chunk[:cut].splitlines() if ln and not ln.startswith(b"host")]
            if rows:
                a = np.array(rows, dtype=float)
                keep = a[:, 0] >= keep_after
                ts.append(a[keep, 0])
                cs.append(a[keep, 1])
        self.t, self.i = np.concatenate(ts), np.concatenate(cs)

    def window(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        m = (self.t >= t0) & (self.t <= t1)
        return self.t[m], self.i[m] * 1e-6


def source_mv() -> int:
    return int(json.loads((STREAM / "status.json").read_text(encoding="utf-8"))["source_mv"])


def stream_alive() -> None:
    st = json.loads((STREAM / "status.json").read_text(encoding="utf-8"))
    age = time.time() - time.mktime(time.strptime(st["time"], "%Y-%m-%dT%H:%M:%S"))
    if age > 5 or st.get("blocks_last_s", 0) < 900:
        raise RuntimeError(f"PPK2 stream stale or slow: {st}")


def per_infer_s(rparam: str, rbin: str, hw: int, threads: int) -> float:
    text = H.run_ncnn(rparam, rbin, hw, 10, 200, threads, timeout=300)
    m = re.search(r"latency_us_mean=([\d.]+)", text)
    return float(m.group(1)) / 1e6


def one(stream: Stream, stem: str, model: tuple[str, str], window_s: float, rep: int,
        n_infer: int, args) -> dict:
    tag = H.short_tag(stem)
    rparam, rbin = model
    hw = H.hw_of(stem)
    stream_alive()
    time.sleep(1.0)
    i0 = time.time()
    time.sleep(args.idle_s)
    i1 = time.time()
    H.run_ncnn(rparam, rbin, hw, args.warmup, 1, args.threads, timeout=180)
    a0 = time.time()
    H.run_ncnn(rparam, rbin, hw, 0, n_infer, args.threads, timeout=max(300, int(window_s * 3) + 60))
    a1 = time.time()
    time.sleep(1.8)
    stream.refresh(keep_after=i0 - 1)
    v = source_mv() / 1e3
    ti, ii = stream.window(i0, i1)
    ta, ia = stream.window(a0, a1)
    if len(ti) < 0.9 * args.idle_s * 1000 or len(ta) < 0.9 * (a1 - a0) * 1000:
        raise RuntimeError(f"stream gap: idle {len(ti)} / active {len(ta)} samples")
    p_idle = v * float(np.trapezoid(ii, ti)) / (ti[-1] - ti[0])
    e_act = v * float(np.trapezoid(ia, ta))
    t_act = float(ta[-1] - ta[0])
    bins = ii[: len(ii) // 10 * 10].reshape(-1, 10).mean(1) * v
    rec = {
        "tag": tag, "window_s": window_s, "rep": rep, "n_infer": n_infer,
        "active_s": round(t_act, 4), "n_active_samples": int(len(ta)),
        "idle_s": round(float(ti[-1] - ti[0]), 4), "p_idle_w": round(p_idle, 6),
        "p_idle_sample_std_w": round(float(bins.std()), 6),
        "p_active_w": round(e_act / t_act, 6),
        "energy_mj_per_infer_net": round((e_act - p_idle * t_act) * 1e3 / n_infer, 6),
        "energy_mj_per_infer_gross": round(e_act * 1e3 / n_infer, 6),
        "source_mv": int(v * 1e3),
    }
    print(f"OK {tag}_T{int(window_s)}_r{rep} N={n_infer} act={rec['active_s']}s "
          f"net={rec['energy_mj_per_infer_net']:.4f} Pidle={rec['p_idle_w']:.4f} "
          f"Pact={rec['p_active_w']:.4f}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="2,5,10,30")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    runs_csv = OUT / "runs.csv"
    done = list(csv.DictReader(runs_csv.open(encoding="utf-8"))) if runs_csv.exists() else []
    if not args.summarize_only:
        stream_alive()
        H.lock_perf()
        stream = Stream(STREAM)
        stream.refresh(keep_after=time.time())
        windows = [float(w) for w in args.windows.split(",")]
        anchors = [a for a in args.anchors.split(",") if a]
        models, t_inf = {}, {}
        for stem in anchors:
            models[stem] = H.ensure_ncnn(stem, "int8")
            t_inf[stem] = per_infer_s(*models[stem], H.hw_of(stem), args.threads)
            print(f"{H.short_tag(stem)}: {t_inf[stem]*1e3:.3f} ms/inf", flush=True)
        jobs = [(s, w, r) for s in anchors for w in windows for r in range(1, args.reps + 1)]
        random.Random(7).shuffle(jobs)
        keys = {(d["tag"], float(d["window_s"]), int(d["rep"])) for d in done}
        for stem, w, r in jobs:
            if (H.short_tag(stem), w, r) in keys:
                continue
            n = max(1, round(w / t_inf[stem]))
            try:
                done.append(one(stream, stem, models[stem], w, r, n, args))
            except Exception as ex:  # noqa: BLE001
                print(f"FAIL {H.short_tag(stem)} T{w} r{r}: {ex}", flush=True)
                continue
            with runs_csv.open("w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=RUN_FIELDS)
                wr.writeheader()
                wr.writerows(done)
    summ = summarize(done)
    summ["protocol"] = {
        "meter": "Nordic PPK2 source-meter 5.0 V, 100 kHz raw, 1 kHz block means (bench.ppk2.hold)",
        "backend": "H618 NCNN INT8, fer_ncnn_bench 4 threads", "governor": "performance",
        "idle_s": args.idle_s, "warmup_outside_window": args.warmup,
        "order": "shuffled, random.Random(7)", "ci": "Student t over reps",
        "idle_sample_sd": "SD of 10 ms bins (comparable to 100 Hz P5S1 samples)",
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for tag, v in summ["by_anchor"].items():
        print(tag, json.dumps(v["idle"]))
        for w, s in v["windows"].items():
            print(f"  T={w:>2}s N={s['n_infer']:>5} samples={s['samples_per_window']:>7} "
                  f"net={s['net_mean']:.4f}+-{s['net_sd']:.4f} CI{s['net_ci95']} cv={s['cv']} "
                  f"rel={s['rel_to_longest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
