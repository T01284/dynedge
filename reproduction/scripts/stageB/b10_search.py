#!/usr/bin/env python3
"""B10/O4: evolutionary search with a FLOPs cost or a measured-model cost table.

The loop is train/search_subnets_flops_ctrl.py unchanged (same random_cfg, mutate, elites,
fitness = val_acc - lambda * cost_norm); only the cost differs:
  --cost flops              model.estimate_flops, normalised between the min and max configs
  --cost table --table T    predicted latency per arch key (json), min-max normalised over T
At the end the archive's non-dominated set over (cost, val_acc) is thinned to at most
--front points evenly spaced in cost rank, and their FER+ test accuracy is computed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.common import make_loaders  # noqa: E402
from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402
from train.search_subnets_flops_ctrl import eval_acc, mutate, random_cfg, set_seed  # noqa: E402


def key_of(c: SubnetConfig) -> str:
    return f"w{c.width:g}_d{''.join(map(str, c.depths))}_s{c.img_size}{'g' if c.gray else 'c'}"


def front_of(cands: list[dict], n: int) -> list[dict]:
    nd, best = [], -1.0
    for r in sorted(cands, key=lambda r: (r["cost"], -r["val_acc"])):
        if r["val_acc"] > best:
            nd.append(r)
            best = r["val_acc"]
    if len(nd) <= n:
        return nd
    idx = sorted({int(round(i)) for i in np.linspace(0, len(nd) - 1, n)})
    return [nd[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--cost", choices=["flops", "table"], required=True)
    ap.add_argument("--table", type=Path, help="json {arch_key: predicted latency}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-cost", type=float, default=0.5)
    ap.add_argument("--front", type=int, default=12)
    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ElasticFERSuperNet(num_classes=8, slimmable=True, arch="v3_wide")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.to(device).eval()

    if args.cost == "flops":
        lo = model.estimate_flops(ElasticFERSuperNet.min_config(), 64)
        hi = model.estimate_flops(ElasticFERSuperNet.max_config(), 112)
        cost_of = lambda c: float(model.estimate_flops(c, c.img_size))  # noqa: E731
    else:
        table = json.loads(args.table.read_text())
        lo, hi = min(table.values()), max(table.values())
        cost_of = lambda c: float(table[key_of(c)])  # noqa: E731

    loaders = make_loaders(args.data_root, batch_size=128, num_workers=4, img_size=112, aug="mild",
                           gray_to_3ch=False)
    pop = [random_cfg(False) for _ in range(args.pop)]
    archive: dict[str, dict] = {}
    for g in range(args.gens):
        for cfg in pop:
            key = key_of(cfg)
            if key in archive:
                continue
            acc = eval_acc(model, loaders["val"], device, cfg)
            cost = cost_of(cfg)
            norm = float(min(1.0, max(0.0, (cost - lo) / max(1e-12, hi - lo))))
            archive[key] = {"key": key, "width": cfg.width, "depths": list(cfg.depths), "img_size": cfg.img_size,
                            "gray": cfg.gray, "val_acc": acc, "cost": cost, "cost_norm": norm,
                            "score": acc - args.lambda_cost * norm}
            print(f"gen{g} {key} acc={acc:.4f} cost={cost:.4g} score={archive[key]['score']:.4f}", flush=True)
        ranked = sorted(archive.values(), key=lambda r: r["score"], reverse=True)
        elites = [SubnetConfig(width=r["width"], depths=tuple(r["depths"]), img_size=r["img_size"], gray=r["gray"])
                  for r in ranked[: args.pop // 2]]
        pop = elites + [mutate(random.choice(elites), False) for _ in range(args.pop - len(elites))]

    front = front_of(list(archive.values()), args.front)
    for r in front:
        cfg = SubnetConfig(width=r["width"], depths=tuple(r["depths"]), img_size=r["img_size"], gray=r["gray"])
        r["test_acc"] = eval_acc(model, loaders["test"], device, cfg)
        print(f"front {r['key']} val={r['val_acc']:.4f} test={r['test_acc']:.4f} cost={r['cost']:.4g}", flush=True)
    out = {"cost": args.cost, "table": str(args.table) if args.table else None, "seed": args.seed,
           "pop": args.pop, "gens": args.gens, "lambda_cost": args.lambda_cost, "cost_lo": lo, "cost_hi": hi,
           "n_eval": len(archive), "candidates": list(archive.values()), "front": front,
           "order": [r["key"] for r in front]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} n_eval={len(archive)} front={len(front)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
