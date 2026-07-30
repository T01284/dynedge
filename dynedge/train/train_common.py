"""Shared utilities for Phase-0 baseline trainers."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from datasets.common import make_loaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate_logits(model, loader, device, crit, logits_fn):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = logits_fn(model, x)
            loss = crit(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}


def build_sched(opt, epochs: int, warmup: int):
    warmup = max(0, warmup)
    if warmup > 0 and warmup < epochs:
        return SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(opt, T_max=epochs - warmup, eta_min=opt.param_groups[0]["lr"] * 0.01),
            ],
            milestones=[warmup],
        )
    return CosineAnnealingLR(opt, T_max=epochs, eta_min=opt.param_groups[0]["lr"] * 0.01)


def train_loop(
    *,
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    args,
    model_name: str,
    train_step_fn,
    eval_logits_fn,
):
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = build_sched(opt, args.epochs, args.warmup_epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    tag = f"_{args.tag}" if args.tag else ""
    best_val = -1.0
    best_path = args.out_dir / f"{model_name}_{args.dataset_name}_seed{args.seed}{tag}_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, running_acc, seen = 0.0, 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                loss, logits = train_step_fn(model, x, y, crit)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = x.size(0)
            running_loss += loss.item() * bs
            running_acc += (logits.argmax(1) == y).float().mean().item() * bs
            seen += bs
            pbar.set_postfix(loss=running_loss / seen, acc=running_acc / seen)

        sched.step()
        train_loss = running_loss / max(1, seen)
        train_acc = running_acc / max(1, seen)
        val = evaluate_logits(model, loaders["val"], device, crit, eval_logits_fn)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "lr": opt.param_groups[0]["lr"],
            "sec": elapsed,
            "imgs_per_sec": seen / max(1e-6, elapsed),
        }
        history.append(row)
        print(
            f"epoch {epoch}: train_acc={train_acc:.4f} val_acc={val['acc']:.4f} "
            f"lr={row['lr']:.5g} {row['imgs_per_sec']:.0f} img/s"
        )
        if val["acc"] > best_val:
            best_val = val["acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val,
                    "args": vars(args),
                    "classes": loaders["classes"],
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    test = evaluate_logits(model, loaders["test"], device, crit, eval_logits_fn)
    result = {
        "model": model_name,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "tag": args.tag,
        "best_val_acc": best_val,
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "aug": args.aug,
        "label_smoothing": args.label_smoothing,
        "img_size": args.img_size,
        "train_len": loaders["train_len"],
        "val_len": loaders["val_len"],
        "test_len": loaders["test_len"],
        "history": history,
    }
    out_json = args.out_dir / f"{model_name}_{args.dataset_name}_seed{args.seed}{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2))
    return result


def make_standard_loaders(args):
    return make_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        aug=args.aug,
    )
