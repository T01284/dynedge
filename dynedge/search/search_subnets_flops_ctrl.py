#!/usr/bin/env python3
"""C1 control: evolutionary search with FLOPs cost on the SAME space as LUT search.

Mirrors ``search_subnets_lut.py`` (width × depths × img/gray), but cost = FLOPs
instead of LUT-predicted latency. Used to show LUT>FLOPs on-device, not just
scatter motivation plots.

Formal LUT window used --pop 40 --gens 25; keep those defaults here.
"""
from __future__ import annotations

import argparse
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

from datasets.common import make_loaders
from models.elastic_mbv3 import (
    DEPTH_CHOICES,
    WIDTH_CHOICES,
    ElasticFERSuperNet,
    SubnetConfig,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    sorted_c = sorted(cands, key=lambda r: r["flops"])
    picks = []
    for q in np.linspace(0, 1, n):
        idx = int(round(q * (len(sorted_c) - 1)))
        lo = max(0, idx - max(1, len(sorted_c) // (n * 2)))
        hi = min(len(sorted_c), idx + max(1, len(sorted_c) // (n * 2)) + 1)
        best = max(sorted_c[lo:hi], key=lambda r: r["val_acc"])
        if best["key"] not in {p["key"] for p in picks}:
            picks.append(best)
    for special in (max(cands, key=lambda r: r["val_acc"]), min(cands, key=lambda r: r["flops"])):
        if special["key"] not in {p["key"] for p in picks}:
            picks.append(special)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset", choices=["ferplus", "expw"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mcu", action="store_true")
    ap.add_argument("--tiers", type=int, default=5)
    ap.add_argument(
        "--lambda-cost",
        type=float,
        default=0.5,
        help="fitness = val_acc - lambda * flops_norm",
    )
    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    num_classes = 8 if args.dataset == "ferplus" else 7
    model = ElasticFERSuperNet(num_classes=num_classes, slimmable=True, arch="v3_wide")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.to(device).eval()

    flops_max = model.estimate_flops(ElasticFERSuperNet.max_config(), 112)
    flops_min = model.estimate_flops(ElasticFERSuperNet.min_config(), 64)
    print(f"[flops_ctrl/{args.dataset}] flops_min={flops_min:.3e} flops_max={flops_max:.3e}")

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
            flops = float(model.estimate_flops(cfg, cfg.img_size))
            flops_norm = (flops - flops_min) / max(1.0, flops_max - flops_min)
            flops_norm = float(min(1.0, max(0.0, flops_norm)))
            score = acc - args.lambda_cost * flops_norm
            archive[key] = {
                "key": key,
                "width": cfg.width,
                "depths": list(cfg.depths),
                "img_size": cfg.img_size,
                "gray": cfg.gray,
                "val_acc": acc,
                "flops": flops,
                "flops_norm": flops_norm,
                "score": score,
            }
            print(
                f"gen{g} {key} acc={acc:.4f} flops={flops:.3e} score={score:.4f}",
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
    out = {
        "dataset": args.dataset,
        "cost": "flops",
        "mcu": args.mcu,
        "n_eval": len(cands),
        "gens": args.gens,
        "pop": args.pop,
        "lambda_cost": args.lambda_cost,
        "flops_min": flops_min,
        "flops_max": flops_max,
        "candidates": cands,
        "tiers": picks,
        "selected": picks,
        "distill_queue": [
            {
                "width": p["width"],
                "depths": ",".join(map(str, p["depths"])),
                "img_size": p["img_size"],
                "gray": p["gray"],
                "dtype": "fp32",
                "val_acc": p["val_acc"],
                "flops": p["flops"],
                "student_act": "relu" if (p["gray"] or p["img_size"] == 64) else "hardswish",
            }
            for p in picks
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out} tiers={len(picks)}")
    for p in picks:
        print(f"  {p['key']} acc={p['val_acc']:.4f} flops={p['flops']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
