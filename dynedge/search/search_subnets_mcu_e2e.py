#!/usr/bin/env python3
"""MCU subnet selection via exhaustive e2e latency truth-table (no op-level LUT).

For each curated (width, depths) @64 gray: val_acc from supernet + latency from
on-device e2e table. Fidelity is definitionally 100%.

  python train/search_subnets_mcu_e2e.py \\
    --ckpt results/supernet/supernet_ferplus_seed42_v4_best.pt \\
    --data-root data/ferplus --dataset ferplus \\
    --truth lut/e2e_mcu_truth.csv \\
    --device esp32p4 \\
    --out results/search_lut/search_ferplus_mcu_p4.json
"""
from __future__ import annotations

# AutoDL ckpts may pickle Linux PosixPath — map before torch.load on Windows only.
# Unconditional PosixPath→WindowsPath breaks Linux (import torch → NotImplementedError).
import pathlib
import sys

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lut"))

from datasets.common import build_transforms, make_loaders
from gen_mcu_e2e_enum import (  # noqa: E402
    MCU_DEVICES,
    coverage_report,
    enumerate_cfgs,
    load_truth,
)
from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig


@torch.no_grad()
def eval_acc(model, loader, device, subnet: SubnetConfig) -> float:
    import torch.nn.functional as F

    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        g = x.mean(dim=1, keepdim=True)
        x = g.expand(-1, 3, -1, -1)
        if x.shape[-1] != subnet.img_size:
            x = F.interpolate(
                x, size=(subnet.img_size, subnet.img_size), mode="bilinear", align_corners=False
            )
        with autocast(enabled=device.type == "cuda"):
            logits = model(x, subnet=subnet, return_all_exits=False)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / max(1, total)


def pick_tiers(cands: list[dict], n: int = 3) -> list[dict]:
    if not cands:
        return []
    ranked = sorted(cands, key=lambda r: r["latency_us"])
    picks = []
    for q in __import__("numpy").linspace(0, 1, n):
        idx = int(round(q * (len(ranked) - 1)))
        # local max-acc near latency quantile
        lo = max(0, idx - max(1, len(ranked) // (n * 2)))
        hi = min(len(ranked), idx + max(1, len(ranked) // (n * 2)) + 1)
        best = max(ranked[lo:hi], key=lambda r: r["val_acc"])
        if best["key"] not in {p["key"] for p in picks}:
            picks.append(best)
    for special in (max(cands, key=lambda r: r["val_acc"]), min(cands, key=lambda r: r["latency_us"])):
        if special["key"] not in {p["key"] for p in picks}:
            picks.append(special)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="ImageFolder root for val (e.g. data/calib/linux112/eval/ferplus) when full split missing",
    )
    ap.add_argument("--dataset", choices=["ferplus", "expw"], required=True)
    ap.add_argument("--truth", type=Path, default=ROOT / "lut/e2e_mcu_truth.csv")
    ap.add_argument("--device", choices=list(MCU_DEVICES) + ["both"], default="both")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tiers", type=int, default=3)
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="do not fail when truth table coverage < 100% (flash-infeasible cfgs)",
    )
    args = ap.parse_args()
    if not args.data_root and not args.val_root:
        raise SystemExit("need --data-root or --val-root")

    truth = load_truth(args.truth)
    cov = coverage_report(truth, args.dataset)
    print(
        f"MCU e2e truth coverage {cov['n_have']}/{cov['n_device_cfg']} ({cov['pct']:.1f}%)"
    )
    if cov["n_missing"]:
        print(
            f"[warn] {cov['n_missing']} device×cfg missing — "
            "run mine_mcu_e2e_batch on lut/mcu_e2e_missing.json first"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    num_classes = 8 if args.dataset == "ferplus" else 7
    model = ElasticFERSuperNet(num_classes=num_classes, slimmable=True, arch="v3_wide")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.to(device).eval()

    if args.val_root:
        ds = tv_datasets.ImageFolder(
            args.val_root,
            transform=build_transforms(False, 112, gray_to_3ch=False),
        )
        val_loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        print(f"val_root={args.val_root} n={len(ds)} classes={len(ds.classes)}")
    else:
        loaders = make_loaders(
            Path(args.data_root),
            batch_size=128,
            num_workers=4,
            img_size=112,
            aug="mild" if args.dataset == "ferplus" else "strong",
        )
        val_loader = loaders["val"]

    cfgs = enumerate_cfgs(args.dataset)
    # eval each unique architecture once
    acc_cache: dict[str, float] = {}
    for c in cfgs:
        if c["key"] in acc_cache:
            continue
        subnet = SubnetConfig(
            width=float(c["width"]),
            depths=tuple(int(x) for x in c["depths"].split(",")),
            img_size=64,
            gray=True,
        )
        acc = eval_acc(model, val_loader, device, subnet)
        acc_cache[c["key"]] = acc
        print(f"acc {c['key']}={acc:.4f}", flush=True)

    devices = list(MCU_DEVICES) if args.device == "both" else [args.device]
    by_dev: dict[str, list] = {d: [] for d in devices}
    for c in cfgs:
        for dev in devices:
            lat = truth.get((dev, float(c["width"]), c["depths"]))
            if lat is None:
                continue
            row = {
                "key": c["key"],
                "device": dev,
                "width": c["width"],
                "depths": c["depths"],
                "img_size": 64,
                "gray": True,
                "dtype": "int8",
                "val_acc": acc_cache[c["key"]],
                "latency_us": lat,
                "latency_ms": lat / 1000.0,
                "pred_us": lat,  # alias: truth is the prediction
                "pred_ms": lat / 1000.0,
                "student_act": "relu",
                "cost_source": "e2e_truth_table",
            }
            by_dev[dev].append(row)

    out = {
        "dataset": args.dataset,
        "method": "exhaustive_e2e_truth_table",
        "fidelity": "100% (definitional — measured wall-clock)",
        "coverage": {k: cov[k] for k in ("n_cfg", "n_device_cfg", "n_have", "n_missing", "pct")},
        "op_wallclock_share": cov["op_wallclock_share"],
        "devices": {},
    }
    for dev, rows in by_dev.items():
        picks = pick_tiers(rows, args.tiers)
        out["devices"][dev] = {
            "n_eval": len(rows),
            "candidates": rows,
            "tiers": picks,
            "distill_queue": [
                {
                    "width": p["width"],
                    "depths": p["depths"],
                    "img_size": 64,
                    "gray": True,
                    "dtype": "int8",
                    "student_act": "relu",
                    "val_acc": p["val_acc"],
                    "pred_ms": p["pred_ms"],
                    "device": dev,
                }
                for p in picks
            ],
        }
        print(f"=== {dev} tiers ===")
        for p in picks:
            print(f"  {p['key']} acc={p['val_acc']:.4f} {p['pred_ms']:.2f}ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    if cov["n_missing"] and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
