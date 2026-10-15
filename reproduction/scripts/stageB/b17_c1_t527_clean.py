#!/usr/bin/env python3
"""B17: C1 (Tables III/IV) with every LUT arm taken from a search on its own board's table.

H618 rows: tiers of the H618 searches (unchanged points). T527 / FER+: tiers of
search_ferplus_t527 (4 shared with H618 + 2 distilled on the Orin, see b0_t527_ferplus_c1.py).
T527 / ExpW: tiers of the faithful-data T527 search (search_expw_t527_faithful.json, distilled on
the Orin, benched by b17_expw_t527_tiers.py). The tiers of the broken ExpW T527 search (cost
collapsed to the H618 affine bias) leave the LUT arm. FLOPs arms, pairing and statistics are the
A1/A2 ones (scripts/stageA/a1_a2_c1_stats.py).

Outputs: results/stageB/c1_t527_clean/{iso_acc_*.csv, iso_lat_*.csv, pairs_*.csv, summary.json},
         paper/Neurocomputing/figures/fig5_c1_{h618,t527}_{ferplus,expw}.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageA"))
import a1_a2_c1_stats as A  # noqa: E402

S = ROOT / "results/stageB"
OUT = S / "c1_t527_clean"
SEARCHES = {
    ("h618", "ferplus"): A.SEARCH_DIR / "search_lut/search_ferplus_h618.json",
    ("h618", "expw"): A.SEARCH_DIR / "search_lut/search_expw_h618.json",
    ("t527", "ferplus"): S / "search_ferplus_t527.json",
    ("t527", "expw"): S / "search_expw_t527_faithful.json",
}
NEW = [  # (dataset, latency csv, checkpoint json dir, json tag)
    ("ferplus", ROOT / "results/t527_v4_formal_lut_t527_baked/matrix_summary.csv",
     S / "subnets_v4_formal_t527", "v4_formal_lut_t527"),
    ("expw", S / "expw_t527_faithful/t527/matrix_summary.csv",
     S / "subnets_v4_formal_t527", "v4_formal_lut_t527_faithful"),
]


def tiers(path: Path) -> set[str]:
    return {A.norm_key(t["key"]) for t in json.loads(path.read_text(encoding="utf-8"))["tiers"]}


def lut_points(pts: pd.DataFrame) -> pd.DataFrame:
    new = []
    for ds, lat_csv, jdir, tag in NEW:
        for r in csv.DictReader(lat_csv.open(encoding="utf-8")):
            key = A.norm_key(r["model"].split(f"subnet_{ds}_")[1].split("_seed42")[0])
            acc = json.loads((jdir / f"subnet_{ds}_{key}_seed42_{tag}.json").read_text(encoding="utf-8"))["test_acc"]
            new.append({"arm": "lut", "device": "t527", "dataset": ds, "dtype": r["dtype"], "key": key,
                        "lat": float(r["latency_us_p50"]) / 1e3, "acc": acc})
    new = pd.DataFrame(new)
    new = pd.concat([new, new.groupby(["arm", "device", "dataset", "key"], as_index=False)
                     .agg(lat=("lat", "min"), acc=("acc", "first")).assign(dtype="auto")], ignore_index=True)
    rows = []
    for (dev, ds), path in SEARCHES.items():
        want = tiers(path)
        old = pts[(pts.arm == "lut") & (pts.device == dev) & (pts.dataset == ds) & pts.key.isin(want)]
        add = new[(new.device == dev) & (new.dataset == ds) & new.key.isin(want - set(old.key))]
        got = pd.concat([old, add], ignore_index=True)
        for dt in ("int8", "fp32", "auto"):
            missing = want - set(got[got.dtype == dt].key)
            assert not missing, f"{dev}/{ds}/{dt}: no measurement for {missing}"
        rows.append(got)
    return pd.concat(rows, ignore_index=True)


def tables(acc: pd.DataFrame, lat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = acc.assign(cluster=acc.dataset + "|" + acc.lut_key)
    lat = lat.assign(cluster=lat.dataset + "|" + lat.lut_key)
    n = len(acc)
    t1 = A.condition_table(acc, A.summarize_ratio)
    t1.loc[t1.condition.str.startswith("Pooled"), "condition"] = f"Pooled ({n} pairs, cluster bootstrap)"
    u1 = acc.groupby("cluster").agg(ratio=("ratio", lambda s: float(np.exp(np.log(s).mean()))),
                                    same_arch=("same_arch", "first")).reset_index()
    t1 = pd.concat([t1, pd.DataFrame([
        A.summarize_ratio(u1, f"Pooled, {len(u1)} unique LUT subnets (board geo-mean)"),
        A.summarize_ratio(u1[~u1.same_arch], "  excl. identical-architecture pairs"),
        A.summarize_ratio(acc, "Frontier (cheapest FLOPs point with acc>=LUT)", col="ratio_frontier",
                          cluster="cluster"),
    ])], ignore_index=True)
    t2 = A.condition_table(lat, A.summarize_delta)
    t2.loc[t2.condition.str.startswith("Pooled"), "condition"] = f"Pooled ({n} pairs, cluster bootstrap)"
    u2 = lat.groupby("cluster").agg(d_acc_pp=("d_acc_pp", "mean")).reset_index()
    t2 = pd.concat([t2, pd.DataFrame([
        A.summarize_delta(u2, f"Pooled, {len(u2)} unique LUT subnets (board mean)"),
        A.summarize_delta(lat, "Pooled iid (pairs treated as independent)"),
        A.summarize_delta(lat, "Frontier (best FLOPs acc with lat<=LUT)", col="d_acc_frontier_pp",
                          cluster="cluster"),
    ])], ignore_index=True)
    return t1, t2


def plot(pts: pd.DataFrame) -> None:
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(ROOT / "scripts"))
    from paper_style import H618, T527, TEXT_IN, apply, save

    apply()
    panels = [
        ("h618", "ferplus", "(a) H618, FER+"),
        ("t527", "ferplus", "(b) T527, FER+"),
        ("h618", "expw", "(c) H618, ExpW"),
        ("t527", "expw", "(d) T527, ExpW"),
    ]
    dev_color = {"h618": H618, "t527": T527}
    use = pts[pts.dtype == "int8"]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_IN, 4.15), sharey=True)
    ymin, ymax = 1.0, 0.0
    for ax, (dev, ds, label) in zip(axes.ravel(), panels):
        color = dev_color[dev]
        lut = use[(use.device == dev) & (use.dataset == ds) & (use.arm == "lut")]
        flops = use[(use.device == dev) & (use.dataset == ds) & (use.arm == "flops")]
        for _, r in lut.iterrows():
            j = (flops.acc - r.acc).abs().idxmin()
            ax.plot([r.lat, flops.loc[j, "lat"]], [r.acc, flops.loc[j, "acc"]], color="0.75", lw=0.6, zorder=1)
        ax.scatter(lut.lat, lut.acc, s=26, marker="o", c=color, zorder=3)
        ax.scatter(flops.lat, flops.acc, s=26, marker="^", facecolors="none", edgecolors=color, linewidths=0.8, zorder=3)
        ymin = min(ymin, float(use[(use.device == dev) & (use.dataset == ds)].acc.min()))
        ymax = max(ymax, float(use[(use.device == dev) & (use.dataset == ds)].acc.max()))
        ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=7)
    for ax in axes.ravel():
        ax.set_ylim(ymin - 0.015, ymax + 0.02)
    axes[1, 0].set_xlabel("P50 latency (ms)")
    axes[1, 1].set_xlabel("P50 latency (ms)")
    axes[0, 0].set_ylabel("Test accuracy")
    axes[1, 0].set_ylabel("Test accuracy")
    axes[0, 1].legend(
        handles=[
            Line2D([0], [0], marker="o", color=H618, ls="none", label="LUT"),
            Line2D([0], [0], marker="^", markerfacecolor="none", markeredgecolor=H618, color=H618, ls="none", label="FLOPs"),
        ],
        frameon=False, loc="lower right",
    )
    fig.tight_layout()
    save(fig, "fig5_c1")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pts = A.load_points()
    lut = lut_points(pts)
    new_pts = pd.concat([pts[pts.arm == "flops"], lut], ignore_index=True)
    new_pts.to_csv(OUT / "points.csv", index=False)
    plot(new_pts)
    summary: dict = {"lut_arm": {f"{d}/{s}": sorted(tiers(p)) for (d, s), p in SEARCHES.items()}}
    pd.set_option("display.width", 220)
    for dtype in ("int8", "fp32", "auto"):
        acc, lat = A.pair_rows(new_pts, dtype, {})
        acc.to_csv(OUT / f"pairs_iso_acc_{dtype}.csv", index=False)
        lat.to_csv(OUT / f"pairs_iso_lat_{dtype}.csv", index=False)
        t1, t2 = tables(acc, lat)
        rev, dom = A.reverse_and_dominance(new_pts, dtype)
        rev = rev.assign(cluster=rev.dataset + "|" + rev.flops_key)
        t1 = pd.concat([t1, pd.DataFrame([A.summarize_ratio(
            rev, "FLOPs-anchored frontier", col="ratio_rev", cluster="cluster")])], ignore_index=True)
        t2 = pd.concat([t2, pd.DataFrame([A.summarize_delta(
            rev, "FLOPs-anchored frontier", col="d_acc_rev_pp", cluster="cluster")])], ignore_index=True)
        t1.to_csv(OUT / f"iso_acc_{dtype}.csv", index=False)
        t2.to_csv(OUT / f"iso_lat_{dtype}.csv", index=False)
        summary[dtype] = {"iso_acc": t1.to_dict(orient="records"), "iso_lat": t2.to_dict(orient="records"),
                          "dominance": dom.to_dict(orient="records")}
        print(f"===== {dtype}")
        print(t1[["condition", "n", "median", "geomean", "ci_lo", "ci_hi", "wilcoxon_p"]].round(3).to_string(index=False))
        print(t2[["condition", "n", "median_pp", "ci_lo_pp", "ci_hi_pp", "wilcoxon_p"]].round(3).to_string(index=False))
        print(dom.to_string(index=False))
        if dtype == "int8":
            print(acc[["device", "dataset", "lut_key", "lut_acc", "lut_lat", "flops_key", "flops_acc",
                       "flops_lat", "ratio"]].round(4).to_string(index=False))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
