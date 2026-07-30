"""Phase-2 elastic supernet trainer — sandwich rule + early-exit CE + inplace distill."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.common import make_loaders
from models.elastic_mbv3 import (
    ElasticFERSuperNet,
    SubnetConfig,
    distill_loss,
    exit_loss,
)


class ModelEMA:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def eval_subnet(model, loader, device, subnet: SubnetConfig):
    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(x, subnet=subnet, return_all_exits=False)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / max(1, total)


def sandwich_step(
    model,
    x,
    y,
    crit_weights,
    distill_w: float,
    *,
    ema: ModelEMA | None = None,
    use_ema_teacher: bool = False,
    max_only: bool = False,
):
    """Largest (+ optional min/random); distill toward max or EMA-max final logits.

    max_only: Phase-v3 warmup — train only max_config (no sandwich / distill).
    use_ema_teacher: teacher = EMA shadow max forward (requires ema).
    """
    cfg_max = ElasticFERSuperNet.max_config()

    # EMA teacher must run *before* any autograd-tracked forward: copying EMA
    # weights into `model` would inplace-mutate tensors still held by the graph.
    teacher = None
    if (not max_only) and use_ema_teacher and ema is not None:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            teacher = model(x, subnet=cfg_max, return_all_exits=False).detach()
        model.load_state_dict(backup, strict=False)
        if was_training:
            model.train()

    outs_max = model(x, subnet=cfg_max, return_all_exits=True)
    loss = exit_loss(outs_max, y, crit_weights)

    if max_only:
        return loss, outs_max[-1].detach(), outs_max[-1]

    if teacher is None:
        teacher = outs_max[-1].detach()

    for cfg in (
        ElasticFERSuperNet.min_config(),
        ElasticFERSuperNet.random_config(),
        ElasticFERSuperNet.random_config(),
    ):
        outs = model(x, subnet=cfg, return_all_exits=True)
        loss = loss + exit_loss(outs, y, crit_weights)
        loss = loss + distill_w * distill_loss(outs[-1], teacher)

    return loss, teacher, outs_max[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, required=True, choices=["ferplus", "expw"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--aug", type=str, default="mild", choices=["none", "mild", "strong"])
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument(
        "--max-path-warmup-epochs",
        type=int,
        default=0,
        help="v3: first N epochs train only max_config (before sandwich)",
    )
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA; try 0.999")
    ap.add_argument(
        "--ema-teacher",
        action="store_true",
        help="v3: distill sandwich students toward EMA max logits (needs --ema-decay>0)",
    )
    ap.add_argument("--slimmable", action="store_true", help="v2 channel-slicing + slimmable BN (abandoned)")
    ap.add_argument(
        "--arch",
        type=str,
        default="v1",
        choices=["v1", "v3_wide"],
        help="backbone recipe; v3_wide = Step-B capacity (depth-only)",
    )
    ap.add_argument("--gray-to-3ch", action="store_true", help="MCU gray→3ch dataloader")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--resume", type=Path, default=None, help="full resume (model+opt+epoch)")
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="load weights only and start a fresh run (for v2 continue)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    loaders = make_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        aug=args.aug,
        gray_to_3ch=args.gray_to_3ch,
    )
    use_ema = args.ema_decay > 0
    if args.ema_teacher and not use_ema:
        raise SystemExit("--ema-teacher requires --ema-decay > 0")
    print(
        f"[supernet/{args.dataset_name}] classes={loaders['num_classes']} "
        f"train={loaders['train_len']} img={args.img_size} bs={args.batch_size} "
        f"arch={args.arch} ema={args.ema_decay if use_ema else 0} ema_teacher={args.ema_teacher} "
        f"slimmable={args.slimmable} gray3ch={args.gray_to_3ch} "
        f"max_warmup={args.max_path_warmup_epochs} device={device}"
    )

    model = ElasticFERSuperNet(
        num_classes=loaders["num_classes"],
        slimmable=args.slimmable,
        arch=args.arch,
    ).to(device)
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
    ema = ModelEMA(model, args.ema_decay) if use_ema else None
    exit_w = (0.5, 0.7, 0.9, 1.0)
    grad_clip = 1.0

    start_epoch = 1
    history = []
    best_max_val = -1.0
    ckpt_path = args.out_dir / f"supernet_{args.dataset_name}_seed{args.seed}{tag}_last.pt"
    best_path = args.out_dir / f"supernet_{args.dataset_name}_seed{args.seed}{tag}_best.pt"

    if args.init_from and args.init_from.exists():
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if ema is not None:
            ema = ModelEMA(model, args.ema_decay)
        print(f"init weights from {args.init_from} (fresh optim/sched)")
    elif args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        history = ckpt.get("history", [])
        best_max_val = float(ckpt.get("best_max_val", -1.0))
        if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
            ema.state = {k: v.to(device) for k, v in ckpt["ema"].items()}
        print(
            f"resumed from {args.resume} epoch={start_epoch} "
            f"lr={opt.param_groups[0]['lr']:.5g}"
        )

    def run_eval(loader):
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            acc_max = eval_subnet(model, loader, device, ElasticFERSuperNet.max_config())
            acc_min = eval_subnet(model, loader, device, ElasticFERSuperNet.min_config())
            samples = [ElasticFERSuperNet.random_config() for _ in range(6)]
            acc_rand = [eval_subnet(model, loader, device, c) for c in samples]
            model.load_state_dict(backup, strict=False)
        else:
            acc_max = eval_subnet(model, loader, device, ElasticFERSuperNet.max_config())
            acc_min = eval_subnet(model, loader, device, ElasticFERSuperNet.min_config())
            samples = [ElasticFERSuperNet.random_config() for _ in range(6)]
            acc_rand = [eval_subnet(model, loader, device, c) for c in samples]
        return acc_max, acc_min, acc_rand

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                loss, _, _ = sandwich_step(
                    model,
                    x,
                    y,
                    exit_w,
                    args.distill_weight,
                    ema=ema,
                    use_ema_teacher=args.ema_teacher and epoch > args.max_path_warmup_epochs,
                    max_only=epoch <= args.max_path_warmup_epochs,
                )
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            bs = x.size(0)
            running_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=running_loss / max(1, seen))

        sched.step()
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "lr": opt.param_groups[0]["lr"],
            "sec": time.time() - t0,
            "imgs_per_sec": seen / max(1e-6, time.time() - t0),
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc_max, acc_min, acc_rand = run_eval(loaders["val"])
            row.update(
                {
                    "val_acc_max": acc_max,
                    "val_acc_min": acc_min,
                    "val_acc_rand_mean": float(np.mean(acc_rand)),
                    "val_acc_rand_std": float(np.std(acc_rand)),
                }
            )
            print(
                f"epoch {epoch}: loss={row['train_loss']:.4f} "
                f"val_max={acc_max:.4f} val_min={acc_min:.4f} "
                f"val_rand={row['val_acc_rand_mean']:.4f}±{row['val_acc_rand_std']:.4f} "
                f"{row['imgs_per_sec']:.0f} img/s"
            )
            if acc_max > best_max_val:
                best_max_val = acc_max
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if ema is not None:
                    for k, v in ema.state.items():
                        if k in state:
                            state[k] = v.detach().cpu().clone()
                torch.save(
                    {
                        "model": state,
                        "epoch": epoch,
                        "best_max_val": best_max_val,
                        "args": vars(args),
                        "classes": loaders["classes"],
                        "ema_decay": args.ema_decay,
                    },
                    best_path,
                )
        else:
            print(
                f"epoch {epoch}: loss={row['train_loss']:.4f} "
                f"lr={row['lr']:.5g} {row['imgs_per_sec']:.0f} img/s"
            )

        history.append(row)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch,
                "best_max_val": best_max_val,
                "history": history,
                "args": vars(args),
                "classes": loaders["classes"],
                "ema": ema.state if ema is not None else None,
            },
            ckpt_path,
        )

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    test_max = eval_subnet(model, loaders["test"], device, ElasticFERSuperNet.max_config())
    test_min = eval_subnet(model, loaders["test"], device, ElasticFERSuperNet.min_config())
    result = {
        "model": "elastic_mbv3_supernet",
        "arch": args.arch,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "tag": args.tag,
        "best_val_acc_max": best_max_val,
        "test_acc_max": test_max,
        "test_acc_min": test_min,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "aug": args.aug,
        "ema_decay": args.ema_decay,
        "slimmable": args.slimmable,
        "init_from": str(args.init_from) if args.init_from else None,
        "history": history,
    }
    out_json = args.out_dir / f"supernet_{args.dataset_name}_seed{args.seed}{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
