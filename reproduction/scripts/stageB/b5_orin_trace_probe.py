#!/usr/bin/env python3
"""B5 probe: power profile inside the published Orin P5S1 active windows.

The Orin window wraps a whole `python3 trt_bench.py` call (imports, CUDA context,
engine deserialisation, then N=1000 timed inferences with warmup 0). The timed
inferences occupy only the tail of the window. This splits each window into the
final `tail_s = N * p50` seconds (inference) and everything before it (start-up)
and reports how much of the published net energy each part carries.

Output: results/stageB/b5_energy_window/orin_trace_probe.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/power/orin_p5s1"
LAT = ROOT / "results/orin_trt_matrix.csv"


def main() -> int:
    runs = pd.read_csv(D / "energy_runs.csv")
    lat = pd.read_csv(LAT)
    out = []
    for _, r in runs.iterrows():
        prec = r.backend.split("_")[1]
        m = lat[(lat.model.str.contains(r.tag.replace("w035", "w0.35").replace("w05", "w0.5")
                                         .replace("w075", "w0.75"), regex=False)) &
                (lat.backend.str.endswith(prec))] if "backend" in lat else lat.iloc[0:0]
        p50_ms = float(m.latency_us_p50.iloc[0]) / 1e3 if len(m) else np.nan
        c = pd.read_csv(ROOT / r.csv)
        t = c.host_s.to_numpy(float)
        p = c.power_w.to_numpy(float)
        a0 = t[c["mark"].fillna("").eq("start").to_numpy()][0]
        a1 = t[c["mark"].fillna("").eq("stop").to_numpy()][-1]
        sel = (t >= a0) & (t <= a1)
        ta, pa = t[sel], p[sel]
        pidle = float(r.p_idle_w)
        e_net_total = float(np.trapezoid(pa - pidle, ta))
        tail = r.n_infer * p50_ms / 1e3 if np.isfinite(p50_ms) else np.nan
        cut = a1 - tail if np.isfinite(tail) else np.nan
        tsel = ta >= cut
        e_tail = float(np.trapezoid(pa[tsel] - pidle, ta[tsel])) if tsel.sum() > 2 else np.nan
        out.append({
            "label": f"{r.tag}_{r.backend}_r{r.run}", "active_s": round(a1 - a0, 3),
            "p50_ms": p50_ms, "infer_s": round(tail, 3) if np.isfinite(tail) else None,
            "net_mj_per_inf_published": round(float(r.energy_mj_per_infer_net), 3),
            "net_mj_per_inf_recomputed": round(e_net_total * 1e3 / r.n_infer, 3),
            "net_mj_per_inf_tail_only": round(e_tail * 1e3 / r.n_infer, 3) if np.isfinite(e_tail) else None,
            "p_startup_mean_w": round(float(pa[~tsel].mean()), 3) if (~tsel).any() else None,
            "p_tail_mean_w": round(float(pa[tsel].mean()), 3) if tsel.any() else None,
            "p_idle_w": round(pidle, 3),
        })
    df = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    (ROOT / "results/stageB/b5_energy_window/orin_trace_probe.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
