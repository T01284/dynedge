#!/usr/bin/env python3
"""B10/O3 refit on the K-process re-bench (exploratory, not pre-registered).

Same k=50 / seed-0 training subnets and ridge as the pre-registered O3 model, but the target is
the per-engine minimum over K processes (o1_min_latency.csv) instead of the single O1 run.
Writes the cost table for the exploratory O4b search and held-out tau for both models.

Outputs: results/stageB/orin_sweep/o3m_cost_table_int8.json, o3m_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageA"))
sys.path.insert(0, str(ROOT / "scripts/stageB"))
from a4_predictor import FEATS  # noqa: E402
from b10_o2_o3 import K_SEARCH, SEED_SEARCH, ridge  # noqa: E402
from common import all_subnets, arch_key, features_frame  # noqa: E402

S = ROOT / "results/stageB/orin_sweep"


def main() -> int:
    o1 = pd.read_csv(S / "o1_latency.csv")
    o1 = o1[o1["round"] == 0]
    mn = pd.read_csv(S / "o1_min_latency.csv")
    keys = sorted(mn.arch.unique())
    F = features_frame(keys)
    X = F[FEATS].to_numpy(float)
    fl = F.flops_search.to_numpy()
    tr = np.random.default_rng(SEED_SEARCH).choice(len(keys), K_SEARCH, replace=False)
    te = np.setdiff1d(np.arange(len(keys)), tr)
    prereg = json.loads((S / "o2_o3_summary.json").read_text())["o3_search_model"]["train_keys"]
    assert sorted(prereg) == sorted(keys[i] for i in tr), "training split differs from the pre-registered one"

    summary: dict = {"k": K_SEARCH, "seed": SEED_SEARCH, "n_heldout": int(len(te))}
    models = {}
    for p in ("int8", "fp16"):
        y1 = o1[o1.precision == p].set_index("arch").latency_us_p50.reindex(keys).to_numpy() / 1e3
        ym = mn[mn.precision == p].set_index("arch").latency_us_p50.reindex(keys).to_numpy() / 1e3
        pre = ridge().fit(X[tr], np.log(y1[tr]))
        new = ridge().fit(X[tr], np.log(ym[tr]))
        models[p] = new
        summary[p] = {
            "prereg_vs_single": float(kendalltau(pre.predict(X[te]), y1[te])[0]),
            "prereg_vs_min": float(kendalltau(pre.predict(X[te]), ym[te])[0]),
            "refit_vs_min": float(kendalltau(new.predict(X[te]), ym[te])[0]),
            "flops_vs_single": float(kendalltau(fl[te], y1[te])[0]),
            "flops_vs_min": float(kendalltau(fl[te], ym[te])[0]),
            "refit_alpha": float(new[-1].alpha_),
            "refit_mape": float(np.mean(np.abs(np.exp(new.predict(X[te])) / ym[te] - 1))),
        }
    space = [arch_key(w, d, r) for w, d, r in all_subnets()]
    Fs = features_frame(space)
    pred = np.exp(models["int8"].predict(Fs[FEATS].to_numpy(float)))
    (S / "o3m_cost_table_int8.json").write_text(json.dumps(dict(zip(space, map(float, pred))), indent=1))
    (S / "o3m_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
