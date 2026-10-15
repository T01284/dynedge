#!/usr/bin/env python3
"""B16: proxy fidelity on a second architecture family (ResNet BasicBlock, random weights).

Same statistics as B10/O2: Kendall tau of MACs / params / layer count vs P50 latency (bootstrap
95% CI) and the inversion rate of arch pairs by MACs ratio, per precision. Pairs with identical
MACs are reported in their own bin (ResNet MACs barely depend on how depth is split over stages),
together with the latency spread inside each MACs tie group.

  python scripts/stageB/b16_fam2_analysis.py --device h618
Inputs:  results/stageB/fam2/features.csv, results/stageB/fam2/<device>_latency.csv
Outputs: results/stageB/fam2/<device>_{proxy_tau,inversions}.csv, <device>_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageB"))
from b10_o2_o3 import tau_ci  # noqa: E402

S = ROOT / "results/stageB/fam2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="h618")
    args = ap.parse_args()

    F = pd.read_csv(S / "features.csv").set_index("arch")
    raw = pd.read_csv(S / f"{args.device}_latency.csv")
    raw["lat_ms"] = raw.latency_us_p50 / 1000.0
    summary: dict = {"device": args.device, "n_arch": int(raw.arch.nunique())}

    rows, inv, ties = [], [], []
    for prec, g in raw.groupby("precision"):
        keys = sorted(set(g.arch) & set(F.index))
        lat = g.set_index("arch").lat_ms.reindex(keys).to_numpy()
        f = F.loc[keys]
        for name, col in (("MACs", "macs"), ("params", "params"), ("layers", "n_layers")):
            t, lo, hi = tau_ci(f[col].to_numpy(), lat)
            rows.append({"precision": prec, "proxy": name, "n": len(keys), "tau": t, "ci_lo": lo, "ci_hi": hi})
        mac = f.macs.to_numpy()
        for i, j in combinations(range(len(keys)), 2):
            if mac[i] == mac[j]:
                inv.append({"precision": prec, "macs_ratio": 1.0, "bin": "tie", "inverted": np.nan})
                continue
            a, b = (i, j) if mac[i] < mac[j] else (j, i)
            inv.append({"precision": prec, "macs_ratio": mac[b] / mac[a], "bin": None, "inverted": lat[a] > lat[b]})
        for m, grp in pd.Series(lat, index=keys).groupby(mac):
            if len(grp) > 1:
                ties.append({"precision": prec, "macs": int(m), "n": len(grp),
                             "lat_min_ms": float(grp.min()), "lat_max_ms": float(grp.max()),
                             "spread": float(grp.max() / grp.min())})

    o2 = pd.DataFrame(rows)
    o2.to_csv(S / f"{args.device}_proxy_tau.csv", index=False)
    inv = pd.DataFrame(inv)
    nt = inv.bin != "tie"
    inv.loc[nt, "bin"] = pd.cut(inv.loc[nt, "macs_ratio"], [1.0, 1.10, 1.25, np.inf],
                                labels=["<1.10", "1.10-1.25", ">1.25"], include_lowest=True).astype(str)
    o2b = (inv[nt].assign(inverted=inv[nt].inverted.astype(bool))
           .groupby(["precision", "bin"]).inverted.agg(["size", "sum", "mean"]).reset_index())
    o2b.to_csv(S / f"{args.device}_inversions.csv", index=False)
    ties = pd.DataFrame(ties)
    summary["proxy_tau"] = o2.to_dict(orient="records")
    summary["inversions"] = o2b.to_dict(orient="records")
    summary["tie_pairs"] = {p: int(n) for p, n in inv[~nt].groupby("precision").size().items()}
    if len(ties):
        summary["tie_groups"] = {p: {"n_groups": int(len(g)), "median_spread": float(g.spread.median()),
                                     "max_spread": float(g.spread.max())}
                                 for p, g in ties.groupby("precision")}
        ties.to_csv(S / f"{args.device}_macs_ties.csv", index=False)
    (S / f"{args.device}_summary.json").write_text(json.dumps(summary, indent=2))
    print(o2.to_string(index=False))
    print(o2b.to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("tie_pairs", "tie_groups") if k in summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
