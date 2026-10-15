#!/usr/bin/env python3
"""B0: T527 / FER+ C1 row with the LUT arm taken from the genuine T527 search.

Table III's T527 / FER+ LUT arm reuses the H618 search tiers. The T527 search
(results/stageB/search_ferplus_t527.json) shares four of them and adds
w0.5_d3322@96 and w0.75_d3333@96 (distilled on the Orin, 256x8 accumulation,
benchmarked on the T527 with the same NCNN protocol). This rebuilds the T527 / FER+
row with that arm, using the A1/A2 pairing and statistics unchanged.

Output: results/stageB/b0_t527_ferplus_c1.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageA"))
import a1_a2_c1_stats as A  # noqa: E402

NEW_DIR = ROOT / "results/stageB/subnets_v4_formal_t527"
NEW_LAT = ROOT / "results/t527_v4_formal_lut_t527_baked/matrix_summary.csv"
SEARCH = ROOT / "results/stageB/search_ferplus_t527.json"


def main() -> int:
    tiers = {A.norm_key(t["key"]) for t in json.loads(SEARCH.read_text(encoding="utf-8"))["tiers"]}
    pts = A.load_points()
    base = pts[(pts.device == "t527") & (pts.dataset == "ferplus") & (pts.dtype.isin(["int8", "fp32"]))]
    flops = base[base.arm == "flops"]
    shared = base[(base.arm == "lut") & base.key.isin(tiers)]

    new = []
    for r in csv.DictReader(NEW_LAT.open(encoding="utf-8")):
        key = A.norm_key(r["model"].split("subnet_ferplus_")[1].split("_seed42")[0])
        acc = json.loads((NEW_DIR / f"subnet_ferplus_{key}_seed42_v4_formal_lut_t527.json")
                         .read_text(encoding="utf-8"))["test_acc"]
        new.append({"arm": "lut", "device": "t527", "dataset": "ferplus", "dtype": r["dtype"],
                    "key": key, "lat": float(r["latency_us_p50"]) / 1e3, "acc": acc})
    lut = pd.concat([shared, pd.DataFrame(new)], ignore_index=True)
    missing = tiers - set(lut.key)
    assert not missing, f"tiers without measurements: {missing}"

    out = {"lut_arm": sorted(tiers), "new_tiers": new, "rows": {}}
    prov = {("ferplus", k): {"t527"} for k in tiers}
    for dtype in ("int8", "fp32"):
        sub = pd.concat([flops[flops.dtype == dtype], lut[lut.dtype == dtype]], ignore_index=True)
        acc_pairs, lat_pairs = A.pair_rows(sub, dtype, prov)
        r1 = A.summarize_ratio(acc_pairs, f"T527 / FER+ ({dtype}, T527-searched LUT arm)")
        r2 = A.summarize_delta(lat_pairs, f"T527 / FER+ ({dtype}, T527-searched LUT arm)")
        rev, dom = A.reverse_and_dominance(sub, dtype)
        dom = dom[(dom.device == "t527") & (dom.dataset == "ferplus")]
        out["rows"][dtype] = {
            "iso_acc": r1, "iso_lat": r2,
            "dominance": dom.to_dict(orient="records"),
            "pairs_iso_acc": acc_pairs[["lut_key", "lut_acc", "lut_lat", "flops_key", "flops_acc",
                                        "flops_lat", "ratio"]].to_dict(orient="records"),
            "pairs_iso_lat": lat_pairs[["lut_key", "lut_lat", "lut_acc", "flops_key", "flops_lat",
                                        "flops_acc", "d_acc_pp"]].to_dict(orient="records"),
        }
        print(f"[{dtype}] iso-acc n={r1['n']} median={r1['median']:.3f} CI=[{r1['ci_lo']:.2f},{r1['ci_hi']:.2f}] "
              f"p={r1['wilcoxon_p']:.2f} | iso-lat median={r2['median_pp']:+.2f} pp "
              f"CI=[{r2['ci_lo_pp']:+.2f},{r2['ci_hi_pp']:+.2f}] p={r2['wilcoxon_p']:.2f} | "
              f"dominance {dom.iloc[0].to_dict()}")
        print(acc_pairs[["lut_key", "lut_acc", "lut_lat", "flops_key", "flops_acc", "flops_lat", "ratio"]]
              .round(4).to_string(index=False))
    (ROOT / "results/stageB/b0_t527_ferplus_c1.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
