#!/usr/bin/env python3
"""B10/O2+O3: proxy fidelity and a measured-latency ridge predictor on the Orin sample.

O1 checks: consistency with the 13 B0 architectures (distilled-weight exports) and drift of
the 10 re-benched architectures.
O2: Kendall tau of FLOPs / params / layer count vs P50 latency (bootstrap 95% CI), and the
inversion rate of arch pairs by FLOPs gap, per precision.
O3: ridge on log latency (A4 features, StandardScaler, alpha by LOO over a fixed grid);
learning curve over k with 50 random splits; the pre-registered k=50 (seed 0) INT8 model
predicts all 972 subnets and is written as the cost table for the O4 search.

Outputs: results/stageB/orin_sweep/o2_*.csv, o3_*.csv, o3_cost_table_int8.json, o2_o3_summary.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageA"))
from a4_predictor import FEATS  # noqa: E402
from common import all_subnets, arch_key, features_frame  # noqa: E402

S = ROOT / "results/stageB/orin_sweep"
KS = (5, 10, 20, 30, 50, 100, 150)
N_SPLITS = 50
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
K_SEARCH, SEED_SEARCH = 50, 0


def tau_ci(x, y, n_boot=2000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    t = kendalltau(x, y)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        bs.append(kendalltau(x[i], y[i])[0])
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    return float(t), float(lo), float(hi)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def main() -> int:
    raw = pd.read_csv(S / "o1_latency.csv")
    raw["lat_ms"] = raw.latency_us_p50 / 1000.0
    r0 = raw[raw["round"] == 0]
    summary: dict = {"n_arch": int(r0.arch.nunique())}

    # --- O1 checks
    b0 = pd.read_csv(ROOT / "results/stageB/orin_trt_extended.csv")
    b0["arch"] = b0.arch.str.replace("w1.0_", "w1_")
    m = r0.merge(b0[["arch", "precision", "latency_us_p50"]], on=["arch", "precision"], suffixes=("", "_b0"))
    m["rel"] = m.latency_us_p50 / m.latency_us_p50_b0 - 1
    summary["consistency_b0"] = {p: {"n": int(len(g)), "median_rel_diff": float(g.rel.median()),
                                     "median_abs_rel_diff": float(g.rel.abs().median()),
                                     "max_abs_rel_diff": float(g.rel.abs().max())}
                                 for p, g in m.groupby("precision")}
    rep = raw[raw["round"] == 1].merge(r0, on=["arch", "precision"], suffixes=("_r1", ""))
    if len(rep):
        d = rep.latency_us_p50_r1 / rep.latency_us_p50 - 1
        summary["drift_repeat"] = {"n": int(len(rep)), "median_abs_rel_diff": float(d.abs().median()),
                                   "max_abs_rel_diff": float(d.abs().max())}

    # --- O2
    keys = sorted(r0.arch.unique())
    F = features_frame(keys)
    rows, inv = [], []
    for prec, g in r0.groupby("precision"):
        lat = g.set_index("arch").lat_ms.reindex(keys).to_numpy()
        for name, col in (("FLOPs", "flops_search"), ("params", "params"), ("layers", "n_layers")):
            t, lo, hi = tau_ci(F[col].to_numpy(), lat)
            rows.append({"precision": prec, "proxy": name, "n": len(keys), "tau": t, "ci_lo": lo, "ci_hi": hi})
        fl = F.flops_search.to_numpy()
        for i, j in combinations(range(len(keys)), 2):
            a, b = (i, j) if fl[i] < fl[j] else (j, i)
            ratio = fl[b] / fl[a]
            inv.append({"precision": prec, "flops_ratio": ratio, "inverted": lat[a] > lat[b]})
    o2 = pd.DataFrame(rows)
    o2.to_csv(S / "o2_proxy_tau.csv", index=False)
    inv = pd.DataFrame(inv)
    inv["bin"] = pd.cut(inv.flops_ratio, [1.0, 1.10, 1.25, np.inf], labels=["<1.10", "1.10-1.25", ">1.25"],
                        include_lowest=True)
    o2b = inv.groupby(["precision", "bin"], observed=True).inverted.agg(["size", "sum", "mean"]).reset_index()
    o2b.to_csv(S / "o2_inversions.csv", index=False)
    summary["o2_tau"] = o2.to_dict(orient="records")
    summary["o2_inversions"] = o2b.to_dict(orient="records")

    # --- O3
    X = F[FEATS].to_numpy(float)
    lc = []
    for prec, g in r0.groupby("precision"):
        y = g.set_index("arch").lat_ms.reindex(keys).to_numpy()
        for k in KS:
            if k >= len(keys) - 5:
                continue
            for s in range(N_SPLITS):
                rng = np.random.default_rng(1000 * k + s)
                tr = rng.choice(len(keys), k, replace=False)
                te = np.setdiff1d(np.arange(len(keys)), tr)
                p = np.exp(ridge().fit(X[tr], np.log(y[tr])).predict(X[te]))
                lc.append({"precision": prec, "k": k, "split": s,
                           "tau_ridge": kendalltau(p, y[te])[0],
                           "tau_flops": kendalltau(F.flops_search.to_numpy()[te], y[te])[0],
                           "mape_ridge": float(np.mean(np.abs(p / y[te] - 1)))})
    lc = pd.DataFrame(lc)
    lc.to_csv(S / "o3_learning_curve_raw.csv", index=False)
    agg = lc.groupby(["precision", "k"]).agg(
        tau_ridge=("tau_ridge", "mean"), tau_ridge_sd=("tau_ridge", "std"), tau_flops=("tau_flops", "mean"),
        ridge_beats_flops=("tau_ridge", lambda s: float((s.to_numpy() > lc.loc[s.index, "tau_flops"].to_numpy()).mean())),
        mape_ridge=("mape_ridge", "mean")).reset_index()
    agg.to_csv(S / "o3_learning_curve.csv", index=False)
    summary["o3_learning_curve"] = agg.to_dict(orient="records")

    # pre-registered search model: k=50, seed 0, INT8
    y = r0[r0.precision == "int8"].set_index("arch").lat_ms.reindex(keys).to_numpy()
    tr = np.random.default_rng(SEED_SEARCH).choice(len(keys), K_SEARCH, replace=False)
    model = ridge().fit(X[tr], np.log(y[tr]))
    space = [arch_key(w, d, r) for w, d, r in all_subnets()]
    Fs = features_frame(space)
    pred = np.exp(model.predict(Fs[FEATS].to_numpy(float)))
    (S / "o3_cost_table_int8.json").write_text(json.dumps(dict(zip(space, map(float, pred))), indent=1))
    te = np.setdiff1d(np.arange(len(keys)), tr)
    summary["o3_search_model"] = {
        "k": K_SEARCH, "seed": SEED_SEARCH, "alpha": float(model[-1].alpha_), "train_keys": [keys[i] for i in tr],
        "heldout_tau": float(kendalltau(np.exp(model.predict(X[te])), y[te])[0]),
        "heldout_tau_flops": float(kendalltau(F.flops_search.to_numpy()[te], y[te])[0])}
    (S / "o2_o3_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "o3_learning_curve"}, indent=1)[:4000])
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
