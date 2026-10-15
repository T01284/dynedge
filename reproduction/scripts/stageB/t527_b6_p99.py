#!/usr/bin/env python3
"""B6: true per-inference P50/P90/P99 for the T527 VIPLite NBG anchors.

vpm_run prints, for every loop, the host-side "run time for this network" and the
driver "profile inference time". The first --warmup loops are dropped; percentiles
are taken over the remaining --iters per-inference samples (no loop-max proxy).

Usage:
  python scripts/stageB/t527_b6_p99.py              # 6 FER+ anchors, 50 warmup + 2000
  python scripts/stageB/t527_b6_p99.py --limit 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "t527_energy", ROOT / "deploy/t527/scripts/run_p5s1_energy_matrix.py"
)
E = importlib.util.module_from_spec(_spec)
sys.modules["t527_energy"] = E
assert _spec.loader is not None
_spec.loader.exec_module(E)

OUT = ROOT / "results/stageB/b6_t527_p99" / ("" if E.NBG_DIR.name == "v4_npu" else E.NBG_DIR.name)
RE_RUN = re.compile(r"run time for this network 0: (\d+) us")
RE_PROF = re.compile(r"profile inference time=(\d+)us")


def pct(a: np.ndarray) -> dict:
    return {
        "n": int(a.size),
        "mean_ms": round(float(a.mean()) / 1e3, 4),
        "p50_ms": round(float(np.percentile(a, 50)) / 1e3, 4),
        "p90_ms": round(float(np.percentile(a, 90)) / 1e3, 4),
        "p99_ms": round(float(np.percentile(a, 99)) / 1e3, 4),
        "max_ms": round(float(a.max()) / 1e3, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    E.lock_perf()
    anchors = E.ANCHORS[: args.limit] if args.limit else E.ANCHORS
    loops = args.warmup + args.iters
    summary = {
        "protocol": {
            "board": "t527",
            "backend": "VIPLite NBG uint8 (vpm_run -b 1)",
            "governor": "performance",
            "warmup_dropped": args.warmup,
            "iters": args.iters,
            "time_base": "per-loop; profile = driver inference time, run = host-side vpm_run call",
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "models": {},
    }
    for stem in anchors:
        remote = E.ensure_npu(stem)
        log = f"/tmp/b6_{E.short_tag(stem)}.log"
        code, _, err = E.run(
            f"chmod 666 /dev/vipcore 2>/dev/null || true; cd {remote} && "
            f"vpm_run -s sample.txt -l {loops} -b 1 > {log} 2>&1; echo rc=$?",
            E.HOST,
            timeout=max(300, loops // 20),
        )
        _, text, _ = E.run(f"cat {log}", E.HOST, timeout=120)
        (OUT / f"{E.short_tag(stem)}_vpm_run.log").write_text(text, encoding="utf-8")
        run_us = np.array([int(x) for x in RE_RUN.findall(text)], dtype=float)
        prof_us = np.array([int(x) for x in RE_PROF.findall(text)], dtype=float)
        if run_us.size != loops or prof_us.size != loops:
            raise RuntimeError(f"{stem}: parsed run={run_us.size} prof={prof_us.size}, expected {loops}")
        rec = {
            "profile": pct(prof_us[args.warmup :]),
            "run": pct(run_us[args.warmup :]),
        }
        summary["models"][E.short_tag(stem)] = rec
        print(
            f"{E.short_tag(stem):18s} profile P50/P90/P99/max = "
            f"{rec['profile']['p50_ms']}/{rec['profile']['p90_ms']}/{rec['profile']['p99_ms']}/{rec['profile']['max_ms']} ms | "
            f"run P50/P99 = {rec['run']['p50_ms']}/{rec['run']['p99_ms']} ms",
            flush=True,
        )
    out = OUT / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
