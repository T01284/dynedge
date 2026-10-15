#!/usr/bin/env python3
"""A1 + A2: statistical re-analysis of the C1 LUT-vs-FLOPs comparison (Table III).

A1  iso-accuracy latency ratio (FLOPs/LUT), nearest-neighbour pairing identical to
    scripts/plot_c1_flops_vs_lut.py, plus a Pareto-frontier (step) variant.
A2  iso-latency accuracy delta (LUT - FLOPs), same pairing families.

Per condition and pooled: n, mean, median, geometric mean, bootstrap 95% CI of the median
(10k resamples; pooled = cluster bootstrap over LUT subnets), two-sided Wilcoxon signed-rank
on log-ratio (A1) or on delta (A2), matched-pairs rank-biserial r.

Outputs: results/stageA/a1_*.csv|json, a2_*.csv, fig_a1_forest.{pdf,png}
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common import OUT, ROOT, bootstrap_ci, fmt_ci, rank_biserial  # noqa: E402

N_BOOT = 10000
SEARCH_DIR = ROOT / "results/from_autodl/formal_v4"
N_TEST = {"ferplus": 3153, "expw": 9185}


def norm_key(k: str) -> str:
    return k.replace("w1.0_", "w1_")


def lut_provenance() -> dict[tuple[str, str], set[str]]:
    """(dataset, key) -> set of searches whose tiers contain it."""
    prov: dict[tuple[str, str], set[str]] = {}
    for name in ("search_ferplus_h618", "search_expw_h618", "search_expw_t527"):
        d = json.loads((SEARCH_DIR / "search_lut" / f"{name}.json").read_text(encoding="utf-8"))
        ds = d["dataset"]
        for t in d["tiers"]:
            prov.setdefault((ds, norm_key(t["key"])), set()).add(name.split("_")[-1])
    return prov


def load_points() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "results/c1_flops_vs_lut/points.csv").drop(columns=["key"])
    p = p.rename(columns={"short": "key", "latency_ms_p50": "lat", "plot_acc": "acc"})
    auto = (
        p.groupby(["arm", "device", "dataset", "key"], as_index=False)
        .agg(lat=("lat", "min"), acc=("acc", "first"))
        .assign(dtype="auto")
    )
    return pd.concat([p[["arm", "device", "dataset", "dtype", "key", "lat", "acc"]], auto], ignore_index=True)


def pair_rows(pts: pd.DataFrame, dtype: str, prov) -> tuple[pd.DataFrame, pd.DataFrame]:
    iso_acc, iso_lat = [], []
    for dev in ("h618", "t527"):
        for ds in ("ferplus", "expw"):
            sub = pts[(pts.device == dev) & (pts.dataset == ds) & (pts.dtype == dtype)]
            L = sub[sub.arm == "lut"].sort_values("lat")
            F = sub[sub.arm == "flops"]
            for _, l in L.iterrows():
                src = sorted(prov.get((ds, l.key), set()))
                genuine = "h618" in src
                fa = F.iloc[(F.acc - l.acc).abs().to_numpy().argmin()]
                # frontier variant: cheapest FLOPs point reaching >= LUT accuracy
                reach = F[F.acc >= l.acc]
                front_lat = float(reach.lat.min()) if len(reach) else np.nan
                iso_acc.append({
                    "device": dev, "dataset": ds, "dtype": dtype, "lut_key": l.key, "lut_acc": l.acc,
                    "lut_lat": l.lat, "flops_key": fa.key, "flops_acc": fa.acc, "flops_lat": fa.lat,
                    "ratio": fa.lat / l.lat, "same_arch": fa.key == l.key,
                    "ratio_frontier": front_lat / l.lat if np.isfinite(front_lat) else np.nan,
                    "lut_source": "+".join(src), "genuine_lut_cost": genuine,
                })
                fl = F.iloc[(F.lat - l.lat).abs().to_numpy().argmin()]
                under = F[F.lat <= l.lat]
                front_acc = float(under.acc.max()) if len(under) else np.nan
                iso_lat.append({
                    "device": dev, "dataset": ds, "dtype": dtype, "lut_key": l.key, "lut_lat": l.lat,
                    "lut_acc": l.acc, "flops_key": fl.key, "flops_lat": fl.lat, "flops_acc": fl.acc,
                    "d_acc_pp": 100 * (l.acc - fl.acc), "same_arch": fl.key == l.key,
                    "d_acc_frontier_pp": 100 * (l.acc - front_acc) if np.isfinite(front_acc) else np.nan,
                    "lut_source": "+".join(src), "genuine_lut_cost": genuine,
                })
    return pd.DataFrame(iso_acc), pd.DataFrame(iso_lat)


def reverse_and_dominance(pts: pd.DataFrame, dtype: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Symmetric frontier checks: anchor on FLOPs points, and count strict Pareto dominance both ways."""
    rev, dom = [], []
    for dev in ("h618", "t527"):
        for ds in ("ferplus", "expw"):
            sub = pts[(pts.device == dev) & (pts.dataset == ds) & (pts.dtype == dtype)]
            L = sub[sub.arm == "lut"]
            F = sub[sub.arm == "flops"]
            for _, f in F.iterrows():
                reach = L[L.acc >= f.acc]
                under = L[L.lat <= f.lat]
                rev.append({
                    "device": dev, "dataset": ds, "flops_key": f.key,
                    "ratio_rev": f.lat / float(reach.lat.min()) if len(reach) else np.nan,
                    "d_acc_rev_pp": 100 * (float(under.acc.max()) - f.acc) if len(under) else np.nan,
                })

            def dominated(a, B):
                return bool(((B.lat <= a.lat) & (B.acc >= a.acc) & ((B.lat < a.lat) | (B.acc > a.acc))).any())

            dom.append({
                "device": dev, "dataset": ds, "n_lut": len(L), "n_flops": len(F),
                "flops_dominated_by_lut": int(sum(dominated(f, L) for _, f in F.iterrows())),
                "lut_dominated_by_flops": int(sum(dominated(lp, F) for _, lp in L.iterrows())),
            })
    return pd.DataFrame(rev), pd.DataFrame(dom)


def wilcox_p(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.all(x == 0):
        return float("nan")
    return float(wilcoxon(x, alternative="two-sided").pvalue)


def cluster_boot_median(df: pd.DataFrame, col: str, cluster: str, log: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy(dtype=float) for _, g in df.groupby(cluster)]
    meds = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in pick])
        meds[b] = np.median(np.log(v)) if log else np.median(v)
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return (float(np.exp(lo)), float(np.exp(hi))) if log else (float(lo), float(hi))


def summarize_ratio(df: pd.DataFrame, label: str, col: str = "ratio", cluster: str | None = None) -> dict:
    x = df[col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    lx = np.log(x)
    if cluster:
        lo, hi = cluster_boot_median(df[np.isfinite(df[col])], col, cluster, log=True)
    else:
        lo, hi = bootstrap_ci(lx, n_boot=N_BOOT)
        lo, hi = float(np.exp(lo)), float(np.exp(hi))
    return {
        "condition": label, "n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
        "geomean": float(np.exp(lx.mean())), "ci_lo": lo, "ci_hi": hi,
        "wilcoxon_p": wilcox_p(lx), "rank_biserial_r": rank_biserial(lx),
        "frac_lut_faster": float((x > 1).mean()), "bootstrap": "cluster(LUT subnet)" if cluster else "iid",
    }


def summarize_delta(df: pd.DataFrame, label: str, col: str = "d_acc_pp", cluster: str | None = None) -> dict:
    x = df[col].to_numpy(dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    if cluster:
        lo, hi = cluster_boot_median(df[ok], col, cluster, log=False)
    else:
        lo, hi = bootstrap_ci(x, n_boot=N_BOOT)
    return {
        "condition": label, "n": int(len(x)), "mean_pp": float(x.mean()), "median_pp": float(np.median(x)),
        "ci_lo_pp": lo, "ci_hi_pp": hi, "wilcoxon_p": wilcox_p(x), "rank_biserial_r": rank_biserial(x),
        "frac_lut_better": float((x > 0).mean()),
    }


def condition_table(pairs: pd.DataFrame, fn, **kw) -> pd.DataFrame:
    rows = []
    for (dev, ds), g in pairs.groupby(["device", "dataset"], sort=False):
        rows.append(fn(g, f"{dev.upper()} / {'FER+' if ds == 'ferplus' else 'ExpW'}", **kw))
    pairs = pairs.assign(cluster=pairs.dataset + "|" + pairs.lut_key)
    rows.append(fn(pairs, "Pooled (34 pairs, cluster bootstrap)", cluster="cluster", **kw))
    return pd.DataFrame(rows)


def main() -> int:
    prov = lut_provenance()
    pts = load_points()
    out_json: dict = {}

    all_acc, all_lat = {}, {}
    for dtype in ("int8", "fp32", "auto"):
        a, l = pair_rows(pts, dtype, prov)
        all_acc[dtype], all_lat[dtype] = a, l
        a.to_csv(OUT / f"a1_pairs_{dtype}.csv", index=False)
        l.to_csv(OUT / f"a2_pairs_{dtype}.csv", index=False)

    # sanity: INT8 NN pairing must reproduce the published Table III
    ref = pd.read_csv(ROOT / "results/c1_flops_vs_lut/iso_acc.csv")
    chk = all_acc["int8"].merge(ref, left_on=["device", "dataset", "lut_key"],
                                right_on=["device", "dataset", "lut_short"])
    assert len(chk) == 34 and np.allclose(chk.ratio, chk.lat_ratio_flops_over_lut), "Table III mismatch"

    # ---------------- A1 main table (INT8, as in paper)
    a8 = all_acc["int8"].assign(cluster=lambda d: d.dataset + "|" + d.lut_key)
    t1 = condition_table(all_acc["int8"], summarize_ratio)
    # unique-subnet pooled: geometric mean of the two boards per LUT subnet (n=17 independent)
    uniq = a8.groupby("cluster").agg(ratio=("ratio", lambda s: float(np.exp(np.log(s).mean()))),
                                     same_arch=("same_arch", "first"),
                                     genuine=("genuine_lut_cost", "first")).reset_index()
    t1 = pd.concat([t1, pd.DataFrame([
        summarize_ratio(uniq, "Pooled, 17 unique LUT subnets (board geo-mean)"),
        summarize_ratio(uniq[~uniq.same_arch], "  excl. identical-architecture pairs"),
        summarize_ratio(uniq[uniq.genuine], "  only subnets from a working LUT cost (H618 search)"),
    ])], ignore_index=True)
    t1.to_csv(OUT / "a1_table_int8.csv", index=False)
    out_json["a1_int8"] = t1.to_dict(orient="records")

    # ---------------- robustness: dtype and frontier pairing
    rob = []
    for dtype in ("int8", "fp32", "auto"):
        d = all_acc[dtype].assign(cluster=lambda x: x.dataset + "|" + x.lut_key)
        r = summarize_ratio(d, f"{dtype.upper()} nearest-neighbour", cluster="cluster")
        rob.append(r)
        r2 = summarize_ratio(d, f"{dtype.upper()} frontier (cheapest FLOPs point with acc>=LUT)",
                             col="ratio_frontier", cluster="cluster")
        rob.append(r2)
    rev8, dom8 = reverse_and_dominance(pts, "int8")
    rev8 = rev8.assign(cluster=rev8.dataset + "|" + rev8.flops_key)
    rob.append(summarize_ratio(rev8, "INT8 frontier, FLOPs-anchored (cheapest LUT point with acc>=FLOPs)",
                               col="ratio_rev", cluster="cluster"))
    rob = pd.DataFrame(rob)
    rob.to_csv(OUT / "a1_robustness.csv", index=False)
    dom8.to_csv(OUT / "a1_dominance_int8.csv", index=False)
    out_json["a1_dominance_int8"] = dom8.to_dict(orient="records")
    print(dom8.to_string(index=False))
    out_json["a1_robustness"] = rob.to_dict(orient="records")

    # ---------------- A2 iso-latency accuracy delta
    t2 = condition_table(all_lat["int8"], summarize_delta)
    l8 = all_lat["int8"].assign(cluster=lambda d: d.dataset + "|" + d.lut_key)
    uniq2 = l8.groupby("cluster").agg(d_acc_pp=("d_acc_pp", "mean")).reset_index()
    t2 = pd.concat([t2, pd.DataFrame([
        summarize_delta(uniq2, "Pooled, 17 unique LUT subnets (board mean)")])], ignore_index=True)
    t2f = condition_table(all_lat["int8"], summarize_delta, col="d_acc_frontier_pp")
    t2.to_csv(OUT / "a2_table_int8.csv", index=False)
    t2f.to_csv(OUT / "a2_table_int8_frontier.csv", index=False)
    rob2 = []
    for dtype in ("int8", "fp32", "auto"):
        d = all_lat[dtype].assign(cluster=lambda x: x.dataset + "|" + x.lut_key)
        rob2.append(summarize_delta(d, f"{dtype.upper()} nearest-neighbour", cluster="cluster"))
        rob2.append(summarize_delta(d, f"{dtype.upper()} frontier (best FLOPs acc with lat<=LUT)",
                                    col="d_acc_frontier_pp", cluster="cluster"))
    rob2.append(summarize_delta(rev8, "INT8 frontier, FLOPs-anchored (best LUT acc with lat<=FLOPs)",
                                col="d_acc_rev_pp", cluster="cluster"))
    rob2 = pd.DataFrame(rob2)
    rob2.to_csv(OUT / "a2_robustness.csv", index=False)
    se = {ds: 100 * np.sqrt(2 * p * (1 - p) / N_TEST[ds]) for ds, p in (("ferplus", 0.87), ("expw", 0.72))}
    out_json["a2_int8"] = t2.to_dict(orient="records")
    out_json["a2_int8_frontier"] = t2f.to_dict(orient="records")
    out_json["a2_robustness"] = rob2.to_dict(orient="records")
    out_json["a2_unpaired_test_se_pp"] = se

    # provenance / audit facts
    t527 = json.loads((SEARCH_DIR / "search_lut/search_expw_t527.json").read_text(encoding="utf-8"))
    preds = [c["pred_us"] for c in t527["candidates"]]
    out_json["audit"] = {
        "t527_expw_search_pred_us_unique": len({round(p, 3) for p in preds}),
        "t527_expw_search_pred_us_range": [min(preds), max(preds)],
        "t527_root_cause": "train/search_subnets_lut.py load_lut() keeps only backend=='h618_ncnn' rows and "
                           "reads column p50_us; lut/ops_t527_v4_full.csv uses backend=ncnn_cpu and "
                           "latency_us_p50, so the T527 table is empty and cost == H618 affine bias.",
        "ferplus_t527_search_exists": False,
        "lut_subnet_sources": {f"{k[0]}|{k[1]}": sorted(v) for k, v in prov.items()},
        "outlier_pairs_ratio_gt_1_5": all_acc["int8"][all_acc["int8"].ratio > 1.5][
            ["device", "dataset", "lut_key", "flops_key", "ratio", "lut_source"]].to_dict(orient="records"),
    }
    (OUT / "a1_a2_summary.json").write_text(json.dumps(out_json, indent=2, default=float), encoding="utf-8")

    # ---------------- forest plot (INT8)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    labels = []
    order = list(t1.condition)
    for i, cond in enumerate(order):
        r = t1[t1.condition == cond].iloc[0]
        y = len(order) - 1 - i
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color="k", lw=1.2)
        sig = r["ci_lo"] > 1 or r["ci_hi"] < 1
        ax.plot(r["median"], y, "s", color="C3" if sig else "C0", ms=5, zorder=3)
        short = (cond.replace("Pooled (34 pairs, cluster bootstrap)", "Pooled, 34 pairs")
                 .replace("Pooled, 17 unique LUT subnets (board geo-mean)", "Pooled, unique subnets")
                 .replace("  excl. identical-architecture pairs", "  excl. identical arch")
                 .replace("  only subnets from a working LUT cost (H618 search)", "  working-LUT subnets only"))
        labels.append(f"{short} (n={r['n']})")
    a = all_acc["int8"]
    for i, ((dev, ds), g) in enumerate(a.groupby(["device", "dataset"], sort=False)):
        y = len(order) - 1 - i
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(g))
        ax.scatter(g.ratio, y + jit, s=7, color="0.55", alpha=0.8, zorder=2, lw=0)
    ax.axvline(1.0, color="0.3", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels[::-1], fontsize=6.2)
    ax.set_xlabel("Iso-accuracy latency ratio FLOPs/LUT\n(median, 95% CI)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_a1_forest.{ext}", dpi=300)
        fig.savefig(ROOT / "paper/Neurocomputing/figures" / f"fig_a1_forest.{ext}", dpi=300)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(t1.round(3).to_string(index=False))
    print(rob.round(3).to_string(index=False))
    print(t2.round(3).to_string(index=False))
    print(t2f.round(3).to_string(index=False))
    print(rob2.round(3).to_string(index=False))
    print(json.dumps(out_json["audit"], indent=1, default=float)[:3000])
    print({k: round(v, 3) for k, v in se.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
