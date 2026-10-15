#!/usr/bin/env python3
"""B5 follow-up: per-window fixed overhead and the window lengths behind the published energies.

1) For the T527 window sweep, regress the window's net energy on N (E_win = e*N + c):
   e is the overhead-free per-inference energy, c the fixed per-window cost
   (process launch, model load) that inflates short windows.
2) Inventory active-window length, N and sample count of the published P5S1 runs per device.

Output: results/stageB/b5_energy_window/window_fit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/stageB/b5_energy_window"


def fit(df: pd.DataFrame) -> dict:
    n = df.n_infer.to_numpy(float)
    e_win = df.energy_mj_per_infer_net.to_numpy(float) * n
    r = stats.linregress(n, e_win)
    tcrit = stats.t.ppf(0.975, len(n) - 2)
    return {
        "e_mj_per_infer": round(r.slope, 5),
        "e_ci95": [round(r.slope - tcrit * r.stderr, 5), round(r.slope + tcrit * r.stderr, 5)],
        "c_mj_per_window": round(r.intercept, 2),
        "c_ci95": [round(r.intercept - tcrit * r.intercept_stderr, 2),
                   round(r.intercept + tcrit * r.intercept_stderr, 2)],
        "r2": round(r.rvalue ** 2, 6), "n_windows": int(len(n)),
    }


def inventory() -> dict:
    inv = {}
    for d in sorted((ROOT / "results/power").glob("*_p5s1*")):
        f = d / "energy_runs.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        if "active_s" not in r:
            continue
        g = {
            "n_runs": int(len(r)),
            "n_infer": sorted({int(x) for x in r.n_infer.dropna()}) if "n_infer" in r else None,
            "active_s_min": round(float(r.active_s.min()), 2),
            "active_s_median": round(float(r.active_s.median()), 2),
            "active_s_max": round(float(r.active_s.max()), 2),
            "idle_s_median": round(float(r.idle_s.median()), 2) if "idle_s" in r else None,
            "runs_per_config": int(r.groupby([c for c in ("model", "backend") if c in r]).size().median())
            if "model" in r else None,
        }
        inv[d.name] = g
    return inv


def main() -> int:
    runs = pd.read_csv(OUT / "t527/runs.csv")
    out = {"t527_fit": {}, "published_windows": inventory()}
    for tag, g in runs.groupby("tag"):
        out["t527_fit"][tag] = fit(g)
        print(tag, out["t527_fit"][tag])
    for k, v in out["published_windows"].items():
        print(k, v)
    (OUT / "window_fit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
