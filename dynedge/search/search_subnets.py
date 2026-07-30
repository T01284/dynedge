"""Phase 2.4 — evolutionary subnet search (depth space, FLOPs proxy cost).

Uses supernet weights (v1 or v3_wide). Width fixed at 1.0 (depth-only).
Selects Pareto / tiered configs for subsequent distill fine-tune.
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
from models.elastic_mbv3 import ARCH_SPECS, DEPTH_CHOICES, ElasticFERSuperNet, SubnetConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_acc(model, loader, device, subnet: SubnetConfig) -> float:
    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(x, subnet=subnet, return_all_exits=False)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / max(1, total)


def mutate(cfg: SubnetConfig) -> SubnetConfig:
    depths = list(cfg.depths)
    i = random.randrange(4)
    depths[i] = random.choice(DEPTH_CHOICES)
    return SubnetConfig(width=1.0, depths=tuple(depths))  # type: ignore[arg-type]


def crossover(a: SubnetConfig, b: SubnetConfig) -> SubnetConfig:
    depths = tuple(a.depths[i] if random.random() < 0.5 else b.depths[i] for i in range(4))
    return SubnetConfig(width=1.0, depths=depths)  # type: ignore[arg-type]


def cfg_key(c: SubnetConfig) -> str:
    return "d" + "".join(str(d) for d in c.depths)


def pick_tiers(candidates: list[dict], n_tiers: int = 5) -> list[dict]:
    """Pick n_tiers along FLOPs axis (low→high), prefer higher acc within bin."""
    if not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda r: r["flops"])
    picks = []
    for target_q in np.linspace(0, 1, n_tiers):
        idx = int(round(target_q * (len(sorted_c) - 1)))
        lo = max(0, idx - max(1, len(sorted_c) // (n_tiers * 2)))
        hi = min(len(sorted_c), idx + max(1, len(sorted_c) // (n_tiers * 2)) + 1)
        window = sorted_c[lo:hi]
        best = max(window, key=lambda r: r["val_acc"])
        if best["key"] not in {p["key"] for p in picks}:
            picks.append(best)
    for special in (
        max(candidates, key=lambda r: r["val_acc"]),
        min(candidates, key=lambda r: r["flops"]),
    ):
        if special["key"] not in {p["key"] for p in picks}:
            picks.append(special)
    return picks[: max(n_tiers, len(picks))]


def resolve_arch(ckpt: dict, cli_arch: str) -> str:
    if cli_arch:
        return cli_arch
    args = ckpt.get("args") or {}
    arch = args.get("arch") if isinstance(args, dict) else None
    if arch in ARCH_SPECS:
        return arch
    return "v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, required=True, choices=["ferplus", "expw"])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arch", type=str, default="", choices=["", "v1", "v3_wide"])
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--mutate-rate", type=float, default=0.1)
    ap.add_argument("--lambda-cost", type=float, default=0.5)
    ap.add_argument("--n-tiers", type=int, default=5)
    ap.add_argument("--tag", type=str, default="v1")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaders = make_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        aug="none",
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch = resolve_arch(ckpt, args.arch)
    model = ElasticFERSuperNet(num_classes=loaders["num_classes"], arch=arch).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[search/{args.dataset_name}] arch={arch} tag={args.tag} device={device}")

    flops_max = model.estimate_flops(ElasticFERSuperNet.max_config(), args.img_size)
    flops_min = model.estimate_flops(ElasticFERSuperNet.min_config(), args.img_size)
    print(f"[search/{args.dataset_name}] flops_min={flops_min:.3e} flops_max={flops_max:.3e}")

    pop: list[SubnetConfig] = [
        ElasticFERSuperNet.max_config(),
        SubnetConfig(width=1.0, depths=(2, 2, 2, 2)),
        SubnetConfig(width=1.0, depths=(3, 3, 3, 3)),
        SubnetConfig(width=1.0, depths=(2, 3, 3, 4)),
        SubnetConfig(width=1.0, depths=(4, 3, 2, 2)),
    ]
    while len(pop) < args.pop_size:
        d = tuple(random.choice(DEPTH_CHOICES) for _ in range(4))
        pop.append(SubnetConfig(width=1.0, depths=d))  # type: ignore[arg-type]

    cache: dict[str, dict] = {}

    def evaluate(cfg: SubnetConfig) -> dict:
        k = cfg_key(cfg)
        if k in cache:
            return cache[k]
        flops = model.estimate_flops(cfg, args.img_size)
        acc = eval_acc(model, loaders["val"], device, cfg)
        cost = (flops - flops_min) / max(1.0, flops_max - flops_min)
        fitness = acc - args.lambda_cost * cost
        row = {
            "key": k,
            "width": cfg.width,
            "depths": list(cfg.depths),
            "flops": flops,
            "flops_norm": cost,
            "val_acc": acc,
            "fitness": fitness,
        }
        cache[k] = row
        return row

    history = []
    for gen in range(args.generations):
        scored = [evaluate(c) for c in pop]
        scored.sort(key=lambda r: r["fitness"], reverse=True)
        best = scored[0]
        history.append(
            {
                "gen": gen,
                "best_fitness": best["fitness"],
                "best_acc": best["val_acc"],
                "best_key": best["key"],
                "unique": len(cache),
            }
        )
        print(
            f"gen {gen:02d}: best={best['key']} acc={best['val_acc']:.4f} "
            f"flops={best['flops']:.3e} fit={best['fitness']:.4f} uniq={len(cache)}"
        )
        elites = scored[: max(2, args.pop_size // 5)]
        elite_cfgs = [
            SubnetConfig(width=1.0, depths=tuple(r["depths"])) for r in elites  # type: ignore[arg-type]
        ]
        next_pop = elite_cfgs.copy()
        while len(next_pop) < args.pop_size:
            a, b = random.sample(elite_cfgs, 2) if len(elite_cfgs) >= 2 else (elite_cfgs[0], elite_cfgs[0])
            child = crossover(a, b)
            if random.random() < args.mutate_rate:
                child = mutate(child)
            next_pop.append(child)
        pop = next_pop

    all_rows = sorted(cache.values(), key=lambda r: r["fitness"], reverse=True)
    selected = pick_tiers(all_rows, n_tiers=args.n_tiers)
    seen = set()
    sel_unique = []
    for r in sorted(selected, key=lambda x: x["flops"]):
        if r["key"] not in seen:
            seen.add(r["key"])
            sel_unique.append(r)

    for r in sel_unique:
        cfg = SubnetConfig(width=1.0, depths=tuple(r["depths"]))  # type: ignore[arg-type]
        r["test_acc"] = eval_acc(model, loaders["test"], device, cfg)

    out = {
        "dataset": args.dataset_name,
        "arch": arch,
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "lambda_cost": args.lambda_cost,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "flops_min": flops_min,
        "flops_max": flops_max,
        "selected": sel_unique,
        "top20": all_rows[:20],
        "search_history": history,
        "n_evaluated": len(cache),
    }
    out_path = args.out_dir / f"search_{args.dataset_name}_seed{args.seed}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"selected": sel_unique, "n_evaluated": len(cache)}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
