#!/usr/bin/env python3
"""B10/O4: pre-registered endpoint of the FLOPs vs measured-model search comparison
(criteria as revised 2026-09-30 15:50, before any search was run; see the plan, B10 R1-R5).

Pairing (per seed, FER+ test accuracy): each anchor subnet is paired with the other arm's front
subnet of nearest test accuracy; the pair is kept only if the anchor lies inside the two fronts'
common accuracy interval and |delta acc| <= 0.5 pp. r = lat_FLOPs / lat_model (>1: model faster),
in both anchor directions. Latency = Orin TensorRT P50 from the single O4 session (all front
subnets re-benched there; mean of two passes if the drift rule triggered a second pass).
Statistics: pooled median r; 95% CI by cluster bootstrap over unique anchor architectures (10k);
two-sided Wilcoxon on the per-architecture mean log r.
PASS (INT8) = model-anchored: n_kept >= 15, median >= 1.05, CI lower > 1.00, p < 0.05; and
FLOPs-anchored: not (median < 1 and p < 0.05).
Secondary (reported only): FP16; reach-frontier ratios; iso-latency accuracy gap; per seed;
O1-vs-O4 and start-vs-end drift; measurement cost of the model arm.

Outputs: results/stageB/orin_sweep/o4_pairs.csv, o4_iso_lat.csv, o4_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageA"))
from a1_a2_c1_stats import summarize_delta  # noqa: E402

S = ROOT / "results/stageB/orin_sweep"
SEEDS = (42, 43, 44)
PRECS = ("int8", "fp16")
TOL = 0.005
MIN_PAIRS = 15
MIN_EFFECT = 1.05
DRIFT_TOL = 0.02
N_BOOT = 10000


def read_lat(name: str) -> pd.DataFrame:
    d = pd.read_csv(S / name)
    return d.pivot_table(index="arch", columns="precision", values="latency_us_p50", aggfunc="mean") / 1000.0


def o4_latency() -> tuple[pd.DataFrame, dict]:
    lat = read_lat("o4_latency.csv")
    info: dict = {"passes": 1}
    if (S / "o4_latency_pass2.csv").is_file():
        lat = (lat + read_lat("o4_latency_pass2.csv").reindex(lat.index)) / 2
        info["passes"] = 2
    o1 = read_lat("o1_latency.csv") if (S / "o1_latency.csv").is_file() else None
    if o1 is not None:
        common = lat.index.intersection(o1.index)
        info["o4_vs_o1"] = {p: {"n": int(len(common)),
                                "median_rel_diff": float((lat.loc[common, p] / o1.loc[common, p] - 1).median())}
                            for p in PRECS}
    if (S / "o4_anchor_end.csv").is_file():
        end = read_lat("o4_anchor_end.csv")
        start = read_lat("o4_latency.csv").loc[end.index]
        info["start_vs_end"] = {p: float((end[p] / start[p] - 1).median()) for p in PRECS}
    return lat, info


def load_front(arm: str, seed: int, lat: pd.DataFrame) -> pd.DataFrame:
    d = json.loads((S / "search" / f"{arm}_s{seed}.json").read_text())
    f = pd.DataFrame(d["front"])[["key", "val_acc", "test_acc", "cost"]]
    missing = sorted(set(f.key) - set(lat.index))
    if missing:
        raise SystemExit(f"{arm}_s{seed}: no O4 latency for {missing}")
    for p in PRECS:
        f[f"lat_{p}"] = lat.loc[f.key, p].to_numpy()
    return f.assign(arm=arm, seed=seed)


def pair(A: pd.DataFrame, B: pd.DataFrame, col: str, anchor_is_model: bool, seed: int) -> list[dict]:
    lo = max(A.test_acc.min(), B.test_acc.min())
    hi = min(A.test_acc.max(), B.test_acc.max())
    out = []
    for _, a in A.iterrows():
        b = B.iloc[(B.test_acc - a.test_acc).abs().to_numpy().argmin()]
        reach = B[B.test_acc >= a.test_acc]
        m, f = (a, b) if anchor_is_model else (b, a)
        kept = bool(lo <= a.test_acc <= hi and abs(a.test_acc - b.test_acc) <= TOL)
        rf = (float(reach[col].min()) / a[col] if anchor_is_model else a[col] / float(reach[col].min())) \
            if len(reach) else np.nan
        out.append({"seed": seed, "anchor": "model" if anchor_is_model else "flops", "cluster": a.key,
                    "model_key": m.key, "model_acc": m.test_acc, "model_lat": m[col],
                    "flops_key": f.key, "flops_acc": f.test_acc, "flops_lat": f[col],
                    "d_acc_pp": 100 * (a.test_acc - b.test_acc), "kept": kept,
                    "ratio": f[col] / m[col], "ratio_reach": rf, "same_arch": m.key == f.key})
    return out


def summarize(df: pd.DataFrame, col: str = "ratio", seed: int = 0) -> dict:
    df = df[np.isfinite(df[col])]
    if df.empty:
        return {"n": 0, "n_clusters": 0}
    lx = np.log(df[col].to_numpy(float))
    groups = [np.log(g[col].to_numpy(float)) for _, g in df.groupby("cluster")]
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
                     for _ in range(N_BOOT)])
    lo, hi = np.exp(np.quantile(meds, [0.025, 0.975]))
    cm = np.array([g.mean() for g in groups])
    p = float(wilcoxon(cm).pvalue) if len(cm) >= 2 and np.any(cm != 0) else float("nan")
    return {"n": int(len(lx)), "n_clusters": len(groups), "median": float(np.exp(np.median(lx))),
            "geomean": float(np.exp(lx.mean())), "ci_lo": float(lo), "ci_hi": float(hi), "wilcoxon_p": p,
            "frac_model_faster": float((lx > 0).mean())}


def verdict(fwd: dict, rev: dict) -> tuple[str, dict]:
    checks = {
        "n_kept>=15": fwd.get("n", 0) >= MIN_PAIRS,
        "median>=1.05": fwd.get("median", 0) >= MIN_EFFECT,
        "ci_lo>1.00": fwd.get("ci_lo", 0) > 1.0,
        "p<0.05": fwd.get("wilcoxon_p", 1) < 0.05,
        "reverse_not_flops_sig": not (rev.get("median", 1) < 1.0 and rev.get("wilcoxon_p", 1) < 0.05),
    }
    if all(checks.values()):
        return "PASS", checks
    if not checks["n_kept>=15"]:
        return "FAIL (insufficient common coverage)", checks
    if fwd.get("ci_hi", 2) < 1.0 or not checks["reverse_not_flops_sig"]:
        return "FAIL (FLOPs better)", checks
    return "FAIL", checks


def measurement_cost() -> dict:
    summ = S / "o2_o3_summary.json"
    if not summ.is_file() or not (S / "o1_latency.csv").is_file():
        return {}
    keys = json.loads(summ.read_text())["o3_search_model"]["train_keys"]
    o1 = pd.read_csv(S / "o1_latency.csv")
    g = o1[(o1["round"] == 0) & (o1.precision == "int8") & o1.arch.isin(keys)]
    bench_s = (g.warmup + g.iters) * g.latency_us_mean / 1e6
    return {"n_arch": int(len(g)), "build_h": float(g.build_s.sum() / 3600),
            "bench_h": float(bench_s.sum() / 3600), "note": "INT8 engines only; FLOPs arm costs nothing"}


def read_min(name: str) -> pd.DataFrame:
    d = pd.read_csv(S / name)
    return d.pivot_table(index="arch", columns="precision", values="latency_us_p50", aggfunc="min") / 1000.0


def o4_latency_multi(extra: bool = False) -> tuple[pd.DataFrame, dict]:
    lat = read_min("o4_multi.csv")
    info: dict = {"source": "o4_multi.csv, per-engine min over K processes",
                  "k": int(pd.read_csv(S / "o4_multi.csv").rep.max()) + 1}
    if extra:
        b = read_min("o4b_multi.csv")
        anchors = json.loads((S / "o4b_order.json").read_text())["anchors"]
        info["o4b_anchor_drift"] = {p: {"n": len(anchors),
                                        "median_rel_diff": float((b.loc[anchors, p] / lat.loc[anchors, p] - 1).median()),
                                        "max_abs_rel_diff": float((b.loc[anchors, p] / lat.loc[anchors, p] - 1).abs().max())}
                                    for p in PRECS}
        lat = pd.concat([lat, b.loc[b.index.difference(lat.index)]])
        info["source"] += " + o4b_multi.csv for archs not in o4_multi"
    return lat, info


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", action="store_true",
                    help="sensitivity: latency = min over the K-process re-bench (outputs o4m_*)")
    ap.add_argument("--arm", default="model",
                    help="cost-model arm; 'modelm' = exploratory search with the K-process-min cost model "
                         "(implies --multi, outputs o4b_*)")
    args = ap.parse_args()
    explore = args.arm != "model"
    out = "o4b" if explore else ("o4m" if args.multi else "o4")
    lat, drift = o4_latency_multi(explore) if (args.multi or explore) else o4_latency()
    fronts = {("flops" if a == "flops" else "model", s): load_front(a, s, lat)
              for a in ("flops", args.arm) for s in SEEDS}
    summary: dict = {"criteria": {"tol_pp": 100 * TOL, "min_pairs": MIN_PAIRS, "min_effect": MIN_EFFECT,
                                  "cluster": "unique anchor architecture", "drift_tol": DRIFT_TOL},
                     "front_sizes": {f"{a}_s{s}": len(f) for (a, s), f in fronts.items()},
                     "drift": drift, "measurement_cost_model_arm": measurement_cost()}
    for tag, v in drift.get("o4_vs_o1", {}).items():
        summary.setdefault("flags", []).extend(
            [f"O4 vs O1 {tag} drift {v['median_rel_diff']:+.1%} > 2% (primary uses O4 only)"]
            if abs(v["median_rel_diff"]) > DRIFT_TOL else [])
    all_pairs, all_iso = [], []
    for prec in PRECS:
        col = f"lat_{prec}"
        rows, iso = [], []
        for s in SEEDS:
            M, F = fronts[("model", s)], fronts[("flops", s)]
            rows += pair(M, F, col, True, s) + pair(F, M, col, False, s)
            for _, m in M.iterrows():
                fl = F.iloc[(F[col] - m[col]).abs().to_numpy().argmin()]
                iso.append({"precision": prec, "seed": s, "cluster": m.key, "model_key": m.key,
                            "model_lat": m[col], "model_acc": m.test_acc, "flops_key": fl.key,
                            "flops_lat": fl[col], "flops_acc": fl.test_acc,
                            "d_acc_pp": 100 * (m.test_acc - fl.test_acc)})
        P = pd.DataFrame(rows).assign(precision=prec)
        I = pd.DataFrame(iso)
        all_pairs.append(P)
        all_iso.append(I)
        fwd = P[(P.anchor == "model") & P.kept]
        rev = P[(P.anchor == "flops") & P.kept]
        res = {
            "model_anchored": summarize(fwd),
            "flops_anchored": summarize(rev),
            "coverage": {"model_anchors": int((P.anchor == "model").sum()), "model_kept": int(len(fwd)),
                         "flops_anchors": int((P.anchor == "flops").sum()), "flops_kept": int(len(rev))},
            "untrimmed_model_anchored": summarize(P[P.anchor == "model"]),
            "reach_model_anchored": summarize(P[P.anchor == "model"], col="ratio_reach"),
            "reach_flops_anchored": summarize(P[P.anchor == "flops"], col="ratio_reach"),
            "iso_latency_acc_gap": summarize_delta(I, f"{prec} iso-lat", cluster="cluster"),
            "per_seed": {str(s): summarize(g) for s, g in fwd.groupby("seed")},
            "same_arch_pairs": int(fwd.same_arch.sum()),
        }
        v, checks = verdict(res["model_anchored"], res["flops_anchored"])
        res["verdict" if prec == "int8" else "verdict_secondary_only"] = v
        res["checks"] = checks
        summary[prec] = res
    pd.concat(all_pairs).to_csv(S / f"{out}_pairs.csv", index=False)
    pd.concat(all_iso).to_csv(S / f"{out}_iso_lat.csv", index=False)
    (S / f"{out}_summary.json").write_text(json.dumps(summary, indent=2))
    for prec in PRECS:
        f, r = summary[prec]["model_anchored"], summary[prec]["flops_anchored"]
        tag = summary[prec].get("verdict", summary[prec].get("verdict_secondary_only"))
        print(f"{prec}: model-anchored n={f.get('n')} ({f.get('n_clusters')} archs) median={f.get('median', np.nan):.3f} "
              f"CI [{f.get('ci_lo', np.nan):.3f}, {f.get('ci_hi', np.nan):.3f}] p={f.get('wilcoxon_p', np.nan):.3g}; "
              f"flops-anchored n={r.get('n')} median={r.get('median', np.nan):.3f} p={r.get('wilcoxon_p', np.nan):.3g} -> {tag}")
    print(json.dumps({"drift": drift, "cost": summary["measurement_cost_model_arm"], "flags": summary.get("flags")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
