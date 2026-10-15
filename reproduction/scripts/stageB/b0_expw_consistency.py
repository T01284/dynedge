#!/usr/bin/env python3
"""Re-evaluate the original ExpW search candidates on the rebuilt ExpW cache.

The supernet and the candidate configs are those of the original search, so the
only thing that changes is the face crops. Reports per-candidate val-accuracy
differences and the Kendall tau between the old and new accuracies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.common import make_loaders  # noqa: E402
from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402
from train.search_subnets_lut import eval_acc, set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--search", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ElasticFERSuperNet(num_classes=7, slimmable=True, arch="v3_wide")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.to(device).eval()
    val_loader = make_loaders(args.data_root, batch_size=128, num_workers=4, img_size=112,
                              aug="strong", gray_to_3ch=False)["val"]

    old: dict[str, dict] = {}
    for p in args.search:
        for c in json.loads(p.read_text(encoding="utf-8"))["candidates"]:
            old.setdefault(c["key"], c)

    rows = []
    for i, (key, c) in enumerate(sorted(old.items())):
        cfg = SubnetConfig(width=c["width"], depths=tuple(c["depths"]), img_size=c["img_size"], gray=c["gray"])
        acc = eval_acc(model, val_loader, device, cfg)
        rows.append({"key": key, "val_acc_orig": c["val_acc"], "val_acc_rebuild": acc})
        print(f"{i + 1}/{len(old)} {key} orig={c['val_acc']:.4f} rebuild={acc:.4f}", flush=True)

    a = np.array([r["val_acc_orig"] for r in rows])
    b = np.array([r["val_acc_rebuild"] for r in rows])
    d = b - a
    tau = kendalltau(a, b)[0]
    summary = {
        "n_candidates": len(rows),
        "n_val": len(val_loader.dataset),
        "mean_diff_pp": round(100 * d.mean(), 3),
        "abs_diff_pp_median": round(100 * np.median(np.abs(d)), 3),
        "abs_diff_pp_max": round(100 * np.abs(d).max(), 3),
        "kendall_tau": round(float(tau), 4),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
