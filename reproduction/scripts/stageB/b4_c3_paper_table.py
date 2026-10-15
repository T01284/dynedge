#!/usr/bin/env python3
"""Compact numbers for the paper's C3 table (lambda=0.2, mid-band budget)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageB"))
from b2_controller_exps import load_feed, read_csv  # noqa: E402
from b3_controller_stats import Test, matched_threshold  # noqa: E402

D = ROOT / "results/stageB/controller"
MID = {"h618": 22.42, "t527": 15.99}


def main() -> int:
    T = Test(ROOT / "results/stageB/logits_ferplus_w0.5_d3333_112")
    sweep = read_csv(D / "b2_threshold_sweep.csv")
    thr = torch.load(D / "b2_threshold_decisions.pt")
    runs = [r for p in sorted(D.glob("b2_runs_*.csv")) for r in read_csv(p)]
    table = read_csv(D / "b2_table.csv")
    out = {}
    for dev, B in MID.items():
        cum = load_feed(dev)[1].numpy()
        ff = T.correct(thr[f"{dev}_fixed3"].long().numpy())
        res = {"force_final": {"acc": ff.mean(), "lat": float(cum[-1])}}
        for kind in ("fixB", "randB"):
            rs = [r for r in runs if r["device"] == dev and r["kind"] == kind and float(r["lambda"]) == 0.2
                  and abs(float(r["budget_ms"]) - B) < 1e-6]
            accs = [float(r["acc"]) for r in rs]
            lats = [float(r["lat_ms"]) for r in rs]
            viol = [float(r["viol"]) for r in rs]
            ta, tl = [], []
            for r in rs:
                name = matched_threshold(sweep, dev, float(r["val_lat_ms"]))
                ch = thr[name].long().numpy()
                ta.append(T.correct(ch).mean())
                tl.append(cum[ch].mean())
            row = next(t for t in table if t["device"] == dev and t["kind"] == kind
                       and float(t["lambda"]) == 0.2 and abs(float(t["budget_ms"]) - B) < 1e-6)
            res[kind] = {"acc": np.mean(accs), "acc_sd": np.std(accs, ddof=1), "lat": np.mean(lats),
                         "viol": np.mean(viol), "thr_acc": np.mean(ta), "thr_lat": np.mean(tl),
                         "d_thr": float(row["dacc_vs_thr_latmatched"]), "d_thr_ci": [float(row["dlm_lo"]), float(row["dlm_hi"])],
                         "d_ff": float(row["dacc_vs_final"]), "d_ff_ci": [float(row["dff_lo"]), float(row["dff_hi"])]}
        rb = [r for r in runs if r["device"] == dev and r["kind"] == "randB" and float(r["lambda"]) == 0.2]
        by_b = {}
        for r in rb:
            by_b.setdefault(float(r["budget_ms"]), []).append(float(r["lat_ms"]))
        res["randB_lat_by_budget"] = {b: float(np.mean(v)) for b, v in sorted(by_b.items())}
        out[dev] = res
    print(json.dumps(out, indent=1, default=float))
    (D / "c3_paper_table.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
