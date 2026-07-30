"""ResNet-18 baseline trainer — high GPU util + FERPlus-friendly knobs."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import models
from tqdm import tqdm

from datasets.common import make_loaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, crit):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(x)
            loss = crit(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}


class ModelEMA:
    """Float-parameter EMA; works with torch.compile state_dict keys."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k not in self.state:
                continue
            self.state[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.state.items():
            if k in msd:
                msd[k].copy_(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, required=True, choices=["ferplus", "expw"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--aug", type=str, default="strong", choices=["none", "mild", "strong"])
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA; try 0.999")
    ap.add_argument("--tag", type=str, default="", help="suffix for output files")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    aug = args.aug

    loaders = make_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        aug=aug,
    )
    num_classes = loaders["num_classes"]
    steps_per_epoch = max(1, loaders["train_len"] // args.batch_size)
    use_ema = args.ema_decay > 0
    print(
        f"[{args.dataset_name}] classes={num_classes} "
        f"train={loaders['train_len']} val={loaders['val_len']} test={loaders['test_len']} "
        f"batch={args.batch_size} steps/epoch≈{steps_per_epoch} "
        f"opt={args.optimizer} aug={aug} ls={args.label_smoothing} "
        f"ema={args.ema_decay if use_ema else 0} device={device}"
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    if device.type == "cuda" and not args.no_compile:
        torch.set_float32_matmul_precision("high")
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    ema = ModelEMA(model, args.ema_decay) if use_ema else None

    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == "sgd":
        opt = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup = max(0, args.warmup_epochs)
    if warmup > 0 and warmup < args.epochs:
        sched = SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(opt, T_max=args.epochs - warmup, eta_min=args.lr * 0.01),
            ],
            milestones=[warmup],
        )
    else:
        sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    scaler = GradScaler(enabled=device.type == "cuda")

    history = []
    best_val = -1.0
    best_path = args.out_dir / f"resnet18_{args.dataset_name}_seed{args.seed}{tag}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, running_acc, seen = 0.0, 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy(logits.detach(), y) * bs
            seen += bs
            pbar.set_postfix(loss=running_loss / seen, acc=running_acc / seen)

        sched.step()
        train_loss = running_loss / max(1, seen)
        train_acc = running_acc / max(1, seen)

        # Validate with EMA weights when enabled (better generalization).
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            val = evaluate(model, loaders["val"], device, crit)
            model.load_state_dict(backup, strict=False)
        else:
            val = evaluate(model, loaders["val"], device, crit)

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
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ema is not None:
                for k, v in ema.state.items():
                    if k in state:
                        state[k] = v.detach().clone()
            torch.save(
                {
                    "model": state,
                    "epoch": epoch,
                    "val_acc": best_val,
                    "args": vars(args),
                    "classes": loaders["classes"],
                    "ema_decay": args.ema_decay,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    test = evaluate(model, loaders["test"], device, crit)
    result = {
        "dataset": args.dataset_name,
        "seed": args.seed,
        "tag": args.tag,
        "best_val_acc": best_val,
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "aug": aug,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "img_size": args.img_size,
        "train_len": loaders["train_len"],
        "val_len": loaders["val_len"],
        "test_len": loaders["test_len"],
        "history": history,
    }
    out_json = args.out_dir / f"resnet18_{args.dataset_name}_seed{args.seed}{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
