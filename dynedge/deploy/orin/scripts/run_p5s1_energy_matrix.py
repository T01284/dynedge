#!/usr/bin/env python3
"""Orin P5S1 energy matrix (TensorRT FP16 / INT8).

Mirror of deploy/h618/scripts/run_p5s1_energy_matrix.py.
Meter must be series on Orin DC high-side (~15 V). Protocol: MAXN_SUPER +
jetson_clocks; idle≥5 s; warmup 50 (outside window); active N=1000.

Usage:
  python deploy/orin/scripts/run_p5s1_energy_matrix.py --port SERIAL_PORT --limit 1 --runs 1
  python deploy/orin/scripts/run_p5s1_energy_matrix.py --port SERIAL_PORT
  python deploy/orin/scripts/run_p5s1_energy_matrix.py --prep-only
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[3]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ssh_orin import DEFAULT_HOST, run  # noqa: E402
from build_engines import (  # noqa: E402
    ANCHORS,
    REMOTE,
    REMOTE_ENG,
    build_one,
    ensure_remote_tree,
    lock_clocks,
    push_bench,
)
from bench.p5s1.energy import summarize_capture, write_summary_json  # noqa: E402
from bench.p5s1.meter import DEFAULT_BAUD, Meter  # noqa: E402

HOST = DEFAULT_HOST
OUT = _ROOT / "results/power/orin_p5s1"

CSV_FIELDS = [
    "host_s",
    "t_ms",
    "voltage_v",
    "current_a",
    "power_w",
    "current_ma",
    "power_mw",
    "mark",
]


def hw_of(stem: str) -> int:
    if "img64" in stem or "_s64g_" in stem:
        return 64
    if "img96" in stem or "_s96c_" in stem:
        return 96
    return 112


def short_tag(stem: str) -> str:
    import re

    m = re.search(r"w[\d.]+_d\d+_s\d+[gc]", stem)
    return m.group(0).replace(".", "") if m else stem[-24:]


def precision_of(backend: str) -> str:
    # trt_fp16 / trt_int8
    if backend.endswith("int8"):
        return "int8"
    if backend.endswith("fp32"):
        return "fp32"
    return "fp16"


@dataclass
class Job:
    stem: str
    backend: str  # trt_fp16 | trt_int8
    run_id: int

    @property
    def label(self) -> str:
        return f"{short_tag(self.stem)}_{self.backend}_r{self.run_id}"

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.stem, self.backend, self.run_id)


class CaptureSession:
    def __init__(self, port: str, rate_hz: int = 100) -> None:
        self.port = port
        self.rate_hz = rate_hz
        self.rows: list[dict] = []
        self.marks: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.meta: dict = {}

    def start(self) -> None:
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if self.rows:
                    return
            time.sleep(0.05)
        raise TimeoutError(f"no P5S1 samples on {self.port}")

    def mark(self, name: str) -> float:
        t = time.monotonic()
        with self._lock:
            self.marks.append((t, name))
        print(f"  # mark {name}", flush=True)
        return t

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._th:
            self._th.join(timeout=5.0)
        with self._lock:
            return list(self.rows)

    def _run(self) -> None:
        with Meter(self.port, baud=DEFAULT_BAUD) as m:
            m.start()
            time.sleep(0.05)
            m.set_sample_rate(self.rate_hz)
            try:
                info = m.wait_info(timeout_s=3.0)
                self.meta = {
                    "meter_model": info.model,
                    "meter_version": info.version,
                    "v_range_v": info.voltage_range_v,
                    "i_range_a": info.current_range_a,
                    "sample_rate_hz": info.sample_rate_hz,
                }
            except TimeoutError as e:
                self.meta = {"warn": str(e)}
            pending: list[tuple[float, str]] = []
            for host_t, s in m.iter_samples(duration_s=None):
                if self._stop.is_set():
                    break
                with self._lock:
                    pending.extend(self.marks)
                    self.marks.clear()
                mark = ""
                while pending and pending[0][0] <= host_t:
                    mark = pending.pop(0)[1]
                row = {
                    "host_s": f"{host_t:.6f}",
                    "t_ms": s.t_ms,
                    "voltage_v": f"{s.voltage_v:.9f}",
                    "current_a": f"{s.current_a:.12f}",
                    "power_w": f"{s.power_w:.12f}",
                    "current_ma": f"{s.current_a * 1e3:.6f}",
                    "power_mw": f"{s.power_w * 1e3:.6f}",
                    "mark": mark,
                }
                with self._lock:
                    self.rows.append(row)
            m.stop()


def run_trt(stem: str, precision: str, hw: int, warmup: int, iters: int, timeout: int) -> str:
    eng = f"{REMOTE_ENG}/{stem}_{precision}.plan"
    cmd = (
        f"python3 {REMOTE}/bench/trt_bench.py --engine {eng} --device orin "
        f"--backend trt_{precision} --h {hw} --w {hw} --warmup {warmup} --iters {iters}"
    )
    code, out, err = run(cmd, HOST, timeout=timeout)
    text = (out or "") + (err or "")
    if code != 0:
        raise RuntimeError(f"trt exit={code}\n{text[-1500:]}")
    return text


def write_csv(path: Path, rows: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def summarize_file(csv_path: Path, n_infer: int, label: str) -> dict:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    starts = [float(r["host_s"]) for r in rows if r.get("mark") == "start"]
    stops = [float(r["host_s"]) for r in rows if r.get("mark") == "stop"]
    i0 = [float(r["host_s"]) for r in rows if r.get("mark") == "idle_start"]
    i1 = [float(r["host_s"]) for r in rows if r.get("mark") == "idle_stop"]
    if not (starts and stops and stops[-1] > starts[0]):
        raise RuntimeError(f"missing start/stop marks in {csv_path}")
    idle = (i0[0], i1[-1]) if i0 and i1 and i1[-1] > i0[0] else None
    summary = summarize_capture(
        rows,
        device="orin",
        label=label,
        n_infer=n_infer,
        sample_rate_hz=100,
        active_host_s=(starts[0], stops[-1]),
        idle_host_s=idle,
        notes="orin_p5s1_matrix",
    )
    out_json = csv_path.with_name(csv_path.stem + "_summary.json")
    write_summary_json(summary, out_json)
    return {
        "energy_mj_per_infer_net": summary.energy_mj_per_infer_net,
        "energy_mj_per_infer_gross": summary.energy_mj_per_infer_gross,
        "p_active_w": summary.active.p_mean_w,
        "p_idle_w": summary.idle.p_mean_w if summary.idle else None,
        "active_s": summary.active.duration_s,
        "idle_s": summary.idle.duration_s if summary.idle else None,
        "summary_json": str(out_json.relative_to(_ROOT)).replace("\\", "/"),
    }


def one_job(job: Job, *, port: str, idle_s: float, n_infer: int, warmup: int) -> dict:
    out_csv = OUT / f"{job.label}.csv"
    print(f"\n== {job.label} ==", flush=True)
    sess = CaptureSession(port)
    sess.start()
    board_log = ""
    prec = precision_of(job.backend)
    hw = hw_of(job.stem)
    try:
        time.sleep(1.0)
        sess.mark("idle_start")
        time.sleep(idle_s)
        sess.mark("idle_stop")
        # warmup outside energy window
        run_trt(job.stem, prec, hw, warmup, 1, timeout=180)
        sess.mark("start")
        timeout = max(300, int(n_infer * 0.05) + 120)
        board_log = run_trt(job.stem, prec, hw, 0, n_infer, timeout=timeout)
        sess.mark("stop")
        time.sleep(0.3)
    finally:
        rows = sess.stop()

    meta = {
        **sess.meta,
        "device": "orin",
        "label": job.label,
        "stem": job.stem,
        "backend": job.backend,
        "run_id": job.run_id,
        "n_infer": n_infer,
        "warmup": warmup,
        "idle_s": idle_s,
        "port": port,
        "host": HOST,
        "nvpmodel": "MAXN_SUPER",
        "n_samples": len(rows),
        "board_log_tail": board_log[-1500:],
    }
    write_csv(out_csv, rows, meta)
    stats = summarize_file(out_csv, n_infer, job.label)
    row = {
        "model": job.stem,
        "tag": short_tag(job.stem),
        "backend": job.backend,
        "run": job.run_id,
        "n_infer": n_infer,
        "csv": str(out_csv.relative_to(_ROOT)).replace("\\", "/"),
        **{k: stats[k] for k in stats if k != "summary_json"},
        "summary_json": stats["summary_json"],
    }
    print(
        f"OK {job.label}  net={row['energy_mj_per_infer_net']:.4f} mJ/inf  "
        f"Pact={row['p_active_w']:.3f}W  Pidle={row['p_idle_w']}",
        flush=True,
    )
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r.get("energy_mj_per_infer_net") in (None, ""):
            continue
        groups.setdefault((r["model"], r["backend"]), []).append(r)
    out = []
    for (model, backend), rs in sorted(groups.items()):
        nets = [float(r["energy_mj_per_infer_net"]) for r in rs]
        gross = [float(r["energy_mj_per_infer_gross"]) for r in rs]
        out.append(
            {
                "model": model,
                "tag": rs[0]["tag"],
                "backend": backend,
                "n_runs": len(rs),
                "energy_mj_per_infer_net_mean": round(statistics.mean(nets), 6),
                "energy_mj_per_infer_net_std": round(statistics.pstdev(nets), 6)
                if len(nets) > 1
                else 0.0,
                "energy_mj_per_infer_gross_mean": round(statistics.mean(gross), 6),
                "p_active_w_mean": round(
                    statistics.mean(float(r["p_active_w"]) for r in rs), 6
                ),
                "p_idle_w_mean": round(
                    statistics.mean(
                        float(r["p_idle_w"]) for r in rs if r["p_idle_w"] not in (None, "")
                    ),
                    6,
                ),
            }
        )
    return out


def prep_models(anchors: list[str], backends: list[str]) -> None:
    lock_clocks(HOST)
    ensure_remote_tree(HOST)
    push_bench(HOST)
    for stem in anchors:
        for be in backends:
            build_one(stem, precision_of(be), HOST, workspace_mib=2048)
    print("prep done", flush=True)


def main() -> int:
    global HOST
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", default="SERIAL_PORT")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-infer", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--idle-s", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backends", default="trt_fp16")
    ap.add_argument("--skip-done", action="store_true", default=True)
    ap.add_argument("--no-skip-done", action="store_false", dest="skip_done")
    ap.add_argument("--prep-only", action="store_true")
    args = ap.parse_args()
    HOST = args.host

    OUT.mkdir(parents=True, exist_ok=True)
    anchors = ANCHORS[: args.limit] if args.limit else ANCHORS
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    if args.prep_only:
        prep_models(anchors, backends)
        (OUT / "READY.md").write_text(
            "# Orin P5S1 ready\n\n"
            f"- Engines under `{REMOTE_ENG}/`\n"
            f"- Bench: `{REMOTE}/bench/trt_bench.py`\n"
            "- Meter: Orin DC high-side (~15 V), SERIAL_PORT, ±3.3 A\n"
            "- Run: `python deploy/orin/scripts/run_p5s1_energy_matrix.py --port SERIAL_PORT`\n",
            encoding="utf-8",
        )
        return 0

    detail_csv = OUT / "energy_runs.csv"
    matrix_csv = OUT / "energy_matrix.csv"
    status = OUT / "status.txt"

    done_rows: list[dict] = []
    done_keys: set[tuple[str, str, int]] = set()
    if detail_csv.exists():
        done_rows = list(csv.DictReader(detail_csv.open(encoding="utf-8")))
        for r in done_rows:
            try:
                done_keys.add((r["model"], r["backend"], int(r["run"])))
            except Exception:
                pass

    prep_models(anchors, backends)
    jobs = [
        Job(stem, be, rid)
        for stem in anchors
        for be in backends
        for rid in range(1, args.runs + 1)
    ]

    with status.open("a", encoding="utf-8") as sf:
        sf.write(
            f"\n===== START {time.strftime('%Y-%m-%dT%H:%M:%S')} jobs={len(jobs)} =====\n"
        )
        sf.flush()
        for job in jobs:
            if args.skip_done and job.key in done_keys:
                print(f"SKIP {job.label}", flush=True)
                continue
            try:
                row = one_job(
                    job,
                    port=args.port,
                    idle_s=args.idle_s,
                    n_infer=args.n_infer,
                    warmup=args.warmup,
                )
            except Exception as ex:
                msg = f"FAIL {job.label}: {ex}"
                print(msg, flush=True)
                sf.write(msg + "\n")
                sf.flush()
                continue
            for k, v in list(row.items()):
                if v is None:
                    row[k] = ""
            done_rows.append(row)
            done_keys.add(job.key)
            with detail_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(done_rows[0].keys()))
                w.writeheader()
                w.writerows(done_rows)
            sf.write(f"OK {job.label} net={row['energy_mj_per_infer_net']}\n")
            sf.flush()

    agg = aggregate(done_rows)
    if agg:
        with matrix_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            w.writeheader()
            w.writerows(agg)
    print(f"\nDONE detail={detail_csv} matrix={matrix_csv} rows={len(done_rows)}")
    for a in agg:
        print(
            f"  {a['tag']:18s} {a['backend']:10s}  "
            f"{a['energy_mj_per_infer_net_mean']:.4f} ± {a['energy_mj_per_infer_net_std']:.4f} mJ"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
