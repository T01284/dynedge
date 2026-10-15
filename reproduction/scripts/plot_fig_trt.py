#!/usr/bin/env python3
"""Two-panel TensorRT figure for sec:trtsearch.

(a) FLOPs versus ten-process minimum P50 for the 191 Orin subnets, with the
    H618 NCNN INT8 architectures that were actually timed overlaid.
(b) Held-out Kendall tau of the ridge model versus FLOPs, on the same minimum
    latencies. The band is the 2.5--97.5 percentile over 50 training splits.
    Means are checked against the published o3m_learning_curve.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "stageA"))
sys.path.insert(0, str(ROOT / "scripts" / "stageB"))
from b10_o2_o3 import KS, N_SPLITS, ridge  # noqa: E402
from a4_predictor import FEATS  # noqa: E402
from common import features_frame, measured_pool  # noqa: E402
from paper_style import FLOPS, H618, ORIN, TEXT_IN, apply, save  # noqa: E402

S = ROOT / "results" / "stageB" / "orin_sweep"


def learning_splits(keys, feat, lat: dict[str, np.ndarray]) -> pd.DataFrame:
    cache = S / "o3m_learning_curve_splits.csv"
    if cache.exists():
        return pd.read_csv(cache)
    x = feat[FEATS].to_numpy(float)
    flops = feat.flops_search.to_numpy()
    rows = []
    for prec, y in lat.items():
        for k in KS:
            if k >= len(keys) - 5:
                continue
            for s in range(N_SPLITS):
                rng = np.random.default_rng(1000 * k + s)
                tr = rng.choice(len(keys), k, replace=False)
                te = np.setdiff1d(np.arange(len(keys)), tr)
                pred = np.exp(ridge().fit(x[tr], np.log(y[tr])).predict(x[te]))
                rows.append({
                    "precision": prec, "k": k, "split": s,
                    "tau_ridge": kendalltau(pred, y[te])[0],
                    "tau_flops": kendalltau(flops[te], y[te])[0],
                })
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def main() -> int:
    apply()
    mn = pd.read_csv(S / "o1_min_latency.csv")
    keys = sorted(mn.arch.unique())
    feat = features_frame(keys)
    flops_m = feat.flops_search.to_numpy() / 1e6
    lat = {}
    for prec in ("int8", "fp16"):
        lat[prec] = mn[mn.precision == prec].set_index("arch").latency_us_p50.reindex(keys).to_numpy() / 1e3

    pool = measured_pool()
    h = pool[pool.platform == "H618 NCNN INT8"]
    hf = features_frame(list(h.key))
    h_flops = hf.flops_search.reindex(h.key).to_numpy() / 1e6
    h_lat = h.lat_ms.to_numpy()

    splits = learning_splits(keys, feat, lat)
    published = pd.read_csv(S / "o3m_learning_curve.csv")
    got = splits.groupby(["precision", "k"]).tau_ridge.mean().rename("got")
    chk = published.merge(got.reset_index(), on=["precision", "k"])
    chk["d"] = (chk.tau_ridge - chk.got).abs()
    print("max |mean tau - published|", float(chk.d.max()))
    if float(chk.d.max()) > 0.005:
        raise SystemExit("learning-curve means do not match o3m_learning_curve.csv")

    span = lat["int8"].max() / lat["int8"].min()
    fspan = flops_m.max() / flops_m.min()
    print(f"Orin INT8 n={len(keys)} latency {lat['int8'].min():.3f}-{lat['int8'].max():.3f} ms ({span:.2f}x) FLOPs {fspan:.1f}x")
    print(f"H618 INT8 n={len(h)}")

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.55))

    ax = axes[0]
    ax.scatter(h_flops, h_lat, s=18, marker="D", facecolors="none", edgecolors=H618,
               linewidths=0.7, label=f"H618 INT8 (n={len(h)})", zorder=2)
    ax.scatter(flops_m, lat["fp16"], s=12, marker="o", facecolors="none", edgecolors=ORIN,
               linewidths=0.6, label="Orin FP16", zorder=3)
    ax.scatter(flops_m, lat["int8"], s=12, marker="o", c=ORIN, label="Orin INT8", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FLOPs (M)")
    ax.set_ylabel("P50 latency (ms)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a)", loc="left", fontsize=8, fontweight="bold", pad=2)

    ax = axes[1]
    style = {
        "int8": dict(color=ORIN, ls="-"),
        "fp16": dict(color="#56B4E9", ls="--"),
    }
    for prec, st in style.items():
        g = splits[splits.precision == prec]
        agg = g.groupby("k").tau_ridge.agg(
            mid="mean", lo=lambda s: np.quantile(s, 0.025), hi=lambda s: np.quantile(s, 0.975),
        )
        fl = g.groupby("k").tau_flops.mean()
        ax.fill_between(agg.index, agg.lo, agg.hi, color=st["color"], alpha=0.18, linewidth=0)
        ax.plot(agg.index, agg.mid, marker="o", ms=3.5, label=f"Ridge {prec.upper()}", **st)
        ax.plot(fl.index, fl.to_numpy(), color=FLOPS, ls=st["ls"], lw=0.9, label=f"FLOPs {prec.upper()}")
    ax.set_xlabel("Measured subnets $k$")
    ax.set_ylabel("Held-out Kendall $\\tau$")
    ax.set_ylim(0.3, 1.02)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("(b)", loc="left", fontsize=8, fontweight="bold", pad=2)

    fig.tight_layout()
    save(fig, "fig_trt_search")
    plt.close(fig)
    print("DONE fig_trt_search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
