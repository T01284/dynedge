#!/usr/bin/env python3
"""Val accuracy of the MTCNN-aligned vs bbox-resized parts of the faithful ExpW reconstruction,
from the per-sample correctness saved by b0_expw_align_probe.py (cut = 39,300)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
CUT = 39300
SHOW = ("w1.0_d4342_s112c", "w1.0_d4334_s112c", "w0.75_d3432_s112c", "w0.5_d3334_s96c")


def main() -> int:
    z = np.load(ROOT / "results/stageB/expw_align_probe.npz")
    keys = list(z["keys"])
    al = (~z["recorded_fb"]) & (z["idx"] < CUT)
    C = np.where(al[None, :], z["perimage"], z["resize"])
    ds = datasets.ImageFolder(ROOT / "data/expw_resize_probe/aligned/val")
    y = np.array([t for _, t in ds.samples])
    neu = [i for c, i in ds.class_to_idx.items() if "neutral" in c][0]

    def bal(c, m):
        return float(np.mean([c[m & (y == j)].mean() for j in range(len(ds.classes)) if (m & (y == j)).any()]))

    out = {"cut": CUT, "n_aligned": int(al.sum()), "n_resized": int((~al).sum()),
           "neutral_share": {"aligned": float(np.mean(y[al] == neu)), "resized": float(np.mean(y[~al] == neu))},
           "subnets": {}}
    for k in SHOW:
        i = keys.index(k)
        c = C[i]
        out["subnets"][k] = {"faithful": float(c.mean()), "orig_recorded": float(z["orig"][i]),
                             "aligned": float(c[al].mean()), "resized": float(c[~al].mean()),
                             "aligned_balanced": bal(c, al), "resized_balanced": bal(c, ~al)}
    d = np.array([C[i][al].mean() - C[i][~al].mean() for i in range(len(keys))])
    wide = np.array([not k.startswith("w0.35") for k in keys])
    out["aligned_minus_resized_pp"] = {"median_all": float(100 * np.median(d)),
                                       "median_w>=0.5": float(100 * np.median(d[wide])),
                                       "median_w0.35": float(100 * np.median(d[~wide]))}
    (ROOT / "results/stageB/expw_subset_acc.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
