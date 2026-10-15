#!/usr/bin/env python3
"""Cache frozen-supernet exit logits for controller experiments (B1-B4).

The controller only sees the four exit logits of a frozen backbone, so the
backbone forward is computed once here. Train logits are stored per epoch with
the same mild augmentation train_controller.py uses; val/test are unaugmented.

Output dir layout (resumable, one file per epoch):
  train_e{k:02d}.pt  {"logits": (N,4,C) fp16, "y": (N,) int64}
  val.pt / test.pt   same keys
  meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "lut", ROOT / "train"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from datasets.common import make_loaders
from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig
from train_controller import adapt_input, parse_depths, set_seed


@torch.no_grad()
def dump(backbone, loader, device, subnet, path: Path) -> int:
    logits, ys = [], []
    for x, y in loader:
        x = adapt_input(x.to(device, non_blocking=True), subnet)
        with autocast(enabled=device.type == "cuda"):
            exits = backbone(x, subnet=subnet, return_all_exits=True)
        logits.append(torch.stack([e.float() for e in exits], dim=1).half().cpu())
        ys.append(y)
    blob = {"logits": torch.cat(logits), "y": torch.cat(ys)}
    tmp = path.with_suffix(".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    return int(blob["y"].numel())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--depths", type=str, default="3,3,3,3")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ElasticFERSuperNet(num_classes=8, slimmable=True, arch="v3_wide")
    backbone.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    backbone.to(device).eval()
    subnet = SubnetConfig(
        width=args.width, depths=parse_depths(args.depths), img_size=args.img_size, gray=False
    )
    loaders = make_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=max(112, args.img_size),
        aug="mild",
        gray_to_3ch=False,
    )
    # train loader drops the last partial batch; caching keeps every sample
    train_loader = torch.utils.data.DataLoader(
        loaders["train"].dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("val", "test"):
        p = args.out_dir / f"{split}.pt"
        if not p.is_file():
            t0 = time.time()
            counts[split] = dump(backbone, loaders[split], device, subnet, p)
            print(f"{split}: n={counts[split]} {time.time() - t0:.0f}s", flush=True)
    for e in range(1, args.epochs + 1):
        p = args.out_dir / f"train_e{e:02d}.pt"
        if p.is_file():
            continue
        torch.manual_seed(args.seed * 1000 + e)
        t0 = time.time()
        n = dump(backbone, train_loader, device, subnet, p)
        print(f"train epoch {e}: n={n} {time.time() - t0:.0f}s", flush=True)
    meta = {
        "ckpt": str(args.ckpt),
        "subnet": {"width": args.width, "depths": args.depths, "img_size": args.img_size},
        "epochs": args.epochs,
        "classes": loaders["classes"],
        "n": {"train": loaders["train_len"], "val": loaders["val_len"], "test": loaders["test_len"]},
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done", json.dumps(meta["n"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
