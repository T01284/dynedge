#!/usr/bin/env python3
"""B16 Orin: collapse the K-process fam2 runs into one latency per (arch, precision).

Each Orin trt_bench process lands in a fast or a slow level, so the per-engine minimum over
K processes estimates the fast level. Writes orin_min_latency.csv and orin_median_latency.csv
(inputs to b16_fam2_analysis.py --device orin_min / orin_median) and a convergence table:
Kendall tau(MACs, min over the first k reps) for k = 1..K, plus split-half agreement.

Inputs:  results/stageB/fam2/orin_multi.csv, features.csv
Outputs: results/stageB/fam2/orin_{min,median}_latency.csv, orin_multi_convergence.csv,
         orin_multi_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "results/stageB/fam2"


def main() -> int:
    d = pd.read_csv(S / "orin_multi.csv")
    f = pd.read_csv(S / "features.csv").set_index("arch")
    g = d.groupby(["arch", "precision"]).latency_us_p50
    for name, v in (("min", g.min()), ("median", g.median())):
        v.rename("latency_us_p50").reset_index().to_csv(S / f"orin_{name}_latency.csv", index=False)

    kmax = int(d.rep.max()) + 1
    conv = []
    for k in range(1, kmax + 1):
        m = d[d.rep < k].groupby(["arch", "precision"]).latency_us_p50.min()
        for p in ("fp16", "int8"):
            x = m.xs(p, level=1)
            conv.append({"k": k, "precision": p, "tau_macs": kendalltau(f.loc[x.index, "macs"], x)[0]})
    conv = pd.DataFrame(conv)
    conv.to_csv(S / "orin_multi_convergence.csv", index=False)

    half = kmax // 2
    a = d[d.rep < half].groupby(["arch", "precision"]).latency_us_p50.min()
    b = d[d.rep >= half].groupby(["arch", "precision"]).latency_us_p50.min()
    rel = (b / a - 1).abs() * 100
    mn = g.min()
    second = g.apply(lambda x: sorted(x)[1]) / mn
    summary = {
        "k": kmax,
        "n_runs": len(d),
        "split_half_absdiff_pct": {p: {q: round(float(rel.xs(p, level=1).quantile(q)), 2) for q in (0.5, 0.9, 1.0)}
                                   for p in ("fp16", "int8")},
        "split_half_tau": {p: round(float(kendalltau(a.xs(p, level=1), b.xs(p, level=1))[0]), 3)
                           for p in ("fp16", "int8")},
        "second_lowest_over_min_median": {p: round(float(second.xs(p, level=1).median()), 4)
                                          for p in ("fp16", "int8")},
        "frac_runs_within_5pct_of_min": {p: round(float(x), 3) for p, x in
                                         (d.join(mn.rename("mn"), on=["arch", "precision"])
                                          .eval("latency_us_p50 / mn < 1.05").groupby(d.precision).mean()).items()},
    }
    (S / "orin_multi_summary.json").write_text(json.dumps(summary, indent=1))
    print(conv.pivot(index="k", columns="precision", values="tau_macs").round(3).to_string())
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
