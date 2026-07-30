#!/usr/bin/env python3
"""Evolutionary subnet search using H618 LUT latency (Phase 2.5 formal).

Cost = min over dtype∈{fp32,int8} of LUT affine prediction for that dtype.
dtype is a decision variable (H618 often prefers FP32 — do not hardcode INT8).

Emits distill queue JSON for train_subnet_distill.py.

Usage (on AutoDL with supernet + data):
  python train/search_subnets_lut.py \\
    --ckpt results/supernet/supernet_ferplus_seed42_v4_best.pt \\
    --data-root data/ferplus --dataset ferplus \\
    --lut lut/ops_h618_v4_full.csv --out results/search_lut_ferplus_v4.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lut"))

from datasets.common import make_loaders
from models.elastic_mbv3 import (
    DEPTH_CHOICES,
    WIDTH_CHOICES,
    ElasticFERSuperNet,
    SubnetConfig,
)
from topo_ops import expand_subnet_ops, op_key

# From H618 fidelity fit (lut/FIDELITY_H618.md)
AFFINE = {
    "fp32": (0.8786167764789027, 1187.1673261297146),
    "int8": (0.8055378987527299, 877.7509894135619),
}
DTYPES = ("fp32", "int8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lut(path: Path) -> dict:
    """Load all H618 op rows; keys include dtype via op_key."""
    lut = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok")).lower() not in {"1", "true"}:
                continue
            if r.get("backend") != "h618_ncnn":
                continue
            if r.get("dtype") not in DTYPES:
                continue
            if not r.get("p50_us"):
                continue
            o = dict(r)
            for num in ("cin", "cout", "k", "s", "h", "w", "m", "n", "groups", "width"):
                if o.get(num) in (None, ""):
                    o[num] = None
                else:
                    try:
                        o[num] = float(o[num]) if "." in str(o[num]) else int(float(o[num]))
                    except ValueError:
                        o[num] = None
            o["backend"] = "h618_ncnn"
            lut[op_key(o)] = float(o["p50_us"])
    return lut


def predict_us_dtype(lut: dict, cfg: SubnetConfig, dtype: str) -> float:
    ops = expand_subnet_ops(
        width=cfg.width,
        depths=cfg.depths,
        img_size=cfg.img_size,
        dtype=dtype,
        backend="h618_ncnn",
    )
    total = 0.0
    for o in ops:
        v = lut.get(op_key(o))
        if v is None:
            if o["op"] in {"add", "gap"}:
                cin = int(o.get("cin") or 1)
                hh = int(o.get("h") or 1)
                ww = int(o.get("w") or hh)
                v = max(1.0, cin * hh * ww / 1e5)
            else:
                v = 0.0
        total += v
    a, b = AFFINE[dtype]
    return a * total + b


def predict_us(lut: dict, cfg: SubnetConfig) -> tuple[float, str]:
    """Return (best_pred_us, chosen_dtype) — dtype is a free decision variable."""
    best_us, best_dt = float("inf"), "fp32"
    for dt in DTYPES:
        us = predict_us_dtype(lut, cfg, dt)
        if us < best_us:
            best_us, best_dt = us, dt
    return best_us, best_dt


@torch.no_grad()
def eval_acc(model, loader, device, subnet: SubnetConfig) -> float:
    import torch.nn.functional as F

    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if subnet.gray or subnet.img_size == 64:
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


def random_cfg(mcu: bool) -> SubnetConfig:
    if mcu:
        return SubnetConfig(
            width=0.35,
            depths=tuple(random.choice(DEPTH_CHOICES) for _ in range(4)),  # type: ignore
            img_size=64,
            gray=True,
        )
    img = random.choice((64, 96, 112))
    return SubnetConfig(
        width=float(random.choice(WIDTH_CHOICES)),
        depths=tuple(random.choice(DEPTH_CHOICES) for _ in range(4)),  # type: ignore
        img_size=img,
        gray=(img == 64),
    )


def mutate(cfg: SubnetConfig, mcu: bool) -> SubnetConfig:
    depths = list(cfg.depths)
    depths[random.randrange(4)] = random.choice(DEPTH_CHOICES)
    if mcu:
        return SubnetConfig(width=0.35, depths=tuple(depths), img_size=64, gray=True)  # type: ignore
    w = cfg.width if random.random() < 0.7 else float(random.choice(WIDTH_CHOICES))
    img = cfg.img_size if random.random() < 0.7 else random.choice((64, 96, 112))
    return SubnetConfig(width=w, depths=tuple(depths), img_size=img, gray=(img == 64))  # type: ignore


def cfg_key(c: SubnetConfig) -> str:
    return f"w{c.width}_d{''.join(map(str, c.depths))}_s{c.img_size}{'g' if c.gray else 'c'}"


def pick_tiers(cands: list[dict], n: int = 5) -> list[dict]:
    if not cands:
        return []
    sorted_c = sorted(cands, key=lambda r: r["pred_us"])
    picks = []
    for q in np.linspace(0, 1, n):
        idx = int(round(q * (len(sorted_c) - 1)))
        lo = max(0, idx - max(1, len(sorted_c) // (n * 2)))
        hi = min(len(sorted_c), idx + max(1, len(sorted_c) // (n * 2)) + 1)
        best = max(sorted_c[lo:hi], key=lambda r: r["val_acc"])
        if best["key"] not in {p["key"] for p in picks}:
            picks.append(best)
    for special in (max(cands, key=lambda r: r["val_acc"]), min(cands, key=lambda r: r["pred_us"])):
        if special["key"] not in {p["key"] for p in picks}:
            picks.append(special)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset", choices=["ferplus", "expw"], required=True)
    ap.add_argument("--lut", type=Path, default=ROOT / "lut/ops_h618_v4_full.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mcu", action="store_true", help="restrict to w0.35@64 gray")
    ap.add_argument("--tiers", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=0, help="0=full val")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "int8"],
        help="auto=min latency over fp32/int8 (default; paper H618 prefers FP32)",
    )
    args = ap.parse_args()
    set_seed(args.seed)

    lut = load_lut(args.lut)
    print(f"lut entries={len(lut)} dtype_mode={args.dtype}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    num_classes = 8 if args.dataset == "ferplus" else 7
    model = ElasticFERSuperNet(num_classes=num_classes, slimmable=True, arch="v3_wide")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.to(device).eval()

    loaders = make_loaders(
        Path(args.data_root),
        batch_size=128,
        num_workers=4,
        img_size=112,
        aug="mild" if args.dataset == "ferplus" else "strong",
        gray_to_3ch=False,
    )
    val_loader = loaders["val"]

    pop = [random_cfg(args.mcu) for _ in range(args.pop)]
    archive: dict[str, dict] = {}

    for g in range(args.gens):
        for cfg in pop:
            key = cfg_key(cfg)
            if key in archive:
                continue
            acc = eval_acc(model, val_loader, device, cfg)
            if args.dtype == "auto":
                pred, dt = predict_us(lut, cfg)
            else:
                pred, dt = predict_us_dtype(lut, cfg, args.dtype), args.dtype
            archive[key] = {
                "key": key,
                "width": cfg.width,
                "depths": list(cfg.depths),
                "img_size": cfg.img_size,
                "gray": cfg.gray,
                "dtype": dt,
                "val_acc": acc,
                "pred_us": pred,
                "pred_ms": pred / 1000.0,
                "score": acc - 0.002 * (pred / 1000.0),
            }
            print(
                f"gen{g} {key} acc={acc:.4f} pred={pred/1000:.2f}ms dtype={dt}",
                flush=True,
            )
        ranked = sorted(archive.values(), key=lambda r: r["score"], reverse=True)
        elites = [
            SubnetConfig(
                width=r["width"],
                depths=tuple(r["depths"]),
                img_size=r["img_size"],
                gray=r["gray"],
            )
            for r in ranked[: args.pop // 2]
        ]  # type: ignore
        pop = elites + [
            mutate(random.choice(elites), args.mcu) for _ in range(args.pop - len(elites))
        ]

    cands = list(archive.values())
    picks = pick_tiers(cands, args.tiers)
    dtype_counts: dict[str, int] = {}
    for r in cands:
        dtype_counts[r["dtype"]] = dtype_counts.get(r["dtype"], 0) + 1
    out = {
        "dataset": args.dataset,
        "mcu": args.mcu,
        "n_eval": len(cands),
        "gens": args.gens,
        "dtype_mode": args.dtype,
        "affine": AFFINE,
        "dtype_choice_counts": dtype_counts,
        "candidates": cands,
        "tiers": picks,
        "distill_queue": [
            {
                "width": p["width"],
                "depths": ",".join(map(str, p["depths"])),
                "img_size": p["img_size"],
                "gray": p["gray"],
                "dtype": p["dtype"],
                "val_acc": p["val_acc"],
                "pred_ms": p["pred_ms"],
                "student_act": "relu" if (p["gray"] or p["img_size"] == 64) else "hardswish",
            }
            for p in picks
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out} tiers={len(picks)} dtype_counts={dtype_counts}")
    for p in picks:
        print(
            f"  {p['key']} acc={p['val_acc']:.4f} {p['pred_ms']:.2f}ms dtype={p['dtype']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
