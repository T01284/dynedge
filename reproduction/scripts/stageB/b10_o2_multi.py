#!/usr/bin/env python3
"""B10/O2 re-done on the K-process re-bench of the O1 sample (orin_b10_multi.sh).

Each Orin trt_bench process lands in a fast or a slow level, so O1 (one process per engine)
mixes the two. Here the per-engine minimum over K processes is the latency. Reports Kendall
tau of FLOPs / params / layers (bootstrap CI), inversion rates by FLOPs gap, tau(FLOPs) vs
the number of processes k, split-half agreement, and agreement with the original O1 run.

Inputs:  results/stageB/orin_sweep/o1_multi.csv, o1_latency.csv
Outputs: results/stageB/orin_sweep/o2m_proxy_tau.csv, o2m_inversions.csv, o2m_convergence.csv,
         o1_min_latency.csv, o2m_summary.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageA"))
sys.path.insert(0, str(ROOT / "scripts/stageB"))
from a4_predictor import FEATS  # noqa: E402
from b10_o2_o3 import KS, N_SPLITS, ridge, tau_ci  # noqa: E402
from common import features_frame  # noqa: E402

S = ROOT / "results/stageB/orin_sweep"


def main() -> int:
    d = pd.read_csv(S / "o1_multi.csv")
    kmax = int(d.rep.max()) + 1
    full = d.groupby(["arch", "precision"]).rep.nunique()
    d = d.join(full.rename("n_rep"), on=["arch", "precision"])
    g = d.groupby(["arch", "precision"]).latency_us_p50
    mn = g.min()
    mn.rename("latency_us_p50").reset_index().to_csv(S / "o1_min_latency.csv", index=False)
    keys = sorted(d.arch.unique())
    F = features_frame(keys)
    summary: dict = {"k": kmax, "n_runs": len(d), "n_arch": len(keys),
                     "reps_per_engine_min": int(full.min())}

    rows, inv = [], []
    for prec in ("fp16", "int8"):
        lat = mn.xs(prec, level=1).reindex(keys).to_numpy()
        for name, col in (("FLOPs", "flops_search"), ("params", "params"), ("layers", "n_layers")):
            t, lo, hi = tau_ci(F[col].to_numpy(), lat)
            rows.append({"precision": prec, "proxy": name, "n": len(keys), "tau": t, "ci_lo": lo, "ci_hi": hi})
        fl = F.flops_search.to_numpy()
        for i, j in combinations(range(len(keys)), 2):
            a, b = (i, j) if fl[i] < fl[j] else (j, i)
            inv.append({"precision": prec, "flops_ratio": fl[b] / fl[a], "inverted": lat[a] > lat[b],
                        "inverted_5pct": lat[a] > 1.05 * lat[b]})
    o2 = pd.DataFrame(rows)
    o2.to_csv(S / "o2m_proxy_tau.csv", index=False)
    inv = pd.DataFrame(inv)
    inv["bin"] = pd.cut(inv.flops_ratio, [1.0, 1.10, 1.25, np.inf], labels=["<1.10", "1.10-1.25", ">1.25"],
                        include_lowest=True)
    o2b = inv.groupby(["precision", "bin"], observed=True).agg(
        size=("inverted", "size"), sum=("inverted", "sum"), mean=("inverted", "mean"),
        mean_5pct=("inverted_5pct", "mean")).reset_index()
    o2b.to_csv(S / "o2m_inversions.csv", index=False)
    summary["tau"] = o2.to_dict(orient="records")
    summary["inversions"] = o2b.to_dict(orient="records")
    summary["latency_ms_range"] = {p: [float(mn.xs(p, level=1).min() / 1e3), float(mn.xs(p, level=1).max() / 1e3)]
                                   for p in ("fp16", "int8")}

    X = F[FEATS].to_numpy(float)
    lc = []
    for p in ("fp16", "int8"):
        y = mn.xs(p, level=1).reindex(keys).to_numpy() / 1e3
        for k in KS:
            if k >= len(keys) - 5:
                continue
            for s in range(N_SPLITS):
                rng = np.random.default_rng(1000 * k + s)
                tr = rng.choice(len(keys), k, replace=False)
                te = np.setdiff1d(np.arange(len(keys)), tr)
                pr = np.exp(ridge().fit(X[tr], np.log(y[tr])).predict(X[te]))
                lc.append({"precision": p, "k": k, "split": s, "tau_ridge": kendalltau(pr, y[te])[0],
                           "tau_flops": kendalltau(F.flops_search.to_numpy()[te], y[te])[0],
                           "mape_ridge": float(np.mean(np.abs(pr / y[te] - 1)))})
    lc = pd.DataFrame(lc)
    lca = lc.groupby(["precision", "k"]).agg(
        tau_ridge=("tau_ridge", "mean"), tau_flops=("tau_flops", "mean"), mape_ridge=("mape_ridge", "mean"),
        ridge_beats_flops=("tau_ridge", lambda s: float((s.to_numpy() > lc.loc[s.index, "tau_flops"].to_numpy()).mean()))
    ).reset_index()
    lca.to_csv(S / "o3m_learning_curve.csv", index=False)
    summary["o3m_learning_curve"] = lca.to_dict(orient="records")

    conv = []
    for k in range(1, kmax + 1):
        m = d[d.rep < k].groupby(["arch", "precision"]).latency_us_p50.min()
        for p in ("fp16", "int8"):
            x = m.xs(p, level=1).reindex(keys)
            conv.append({"k": k, "precision": p, "tau_flops": kendalltau(F.flops_search.to_numpy(), x)[0]})
    conv = pd.DataFrame(conv)
    conv.to_csv(S / "o2m_convergence.csv", index=False)

    half = kmax // 2
    a = d[d.rep < half].groupby(["arch", "precision"]).latency_us_p50.min()
    b = d[d.rep >= half].groupby(["arch", "precision"]).latency_us_p50.min()
    rel = ((b / a - 1).abs() * 100).dropna()
    summary["split_half"] = {p: {"tau": round(float(kendalltau(a.xs(p, level=1), b.xs(p, level=1).reindex(
        a.xs(p, level=1).index))[0]), 3), "absdiff_pct_median": round(float(rel.xs(p, level=1).median()), 2),
        "absdiff_pct_p90": round(float(rel.xs(p, level=1).quantile(0.9)), 2)} for p in ("fp16", "int8")}

    o1 = pd.read_csv(S / "o1_latency.csv")
    o1 = o1[o1["round"] == 0].set_index(["arch", "precision"]).latency_us_p50
    r = (o1 / mn.reindex(o1.index) - 1) * 100
    summary["o1_vs_min"] = {p: {"median_pct": round(float(r.xs(p, level=1).median()), 2),
                                "frac_within_5pct": round(float((r.xs(p, level=1) < 5).mean()), 3),
                                "max_pct": round(float(r.xs(p, level=1).max()), 2)} for p in ("fp16", "int8")}
    (S / "o2m_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(o2.round(3).to_string(index=False))
    print(o2b.round(3).to_string(index=False))
    print(conv.pivot(index="k", columns="precision", values="tau_flops").round(3).to_string())
    print(lca.round(3).to_string(index=False))
    print("latency range ms", summary["latency_ms_range"])
    print(json.dumps({k: summary[k] for k in ("k", "n_runs", "reps_per_engine_min", "split_half", "o1_vs_min")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
