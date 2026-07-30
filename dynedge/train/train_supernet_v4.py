"""Phase-2 overnight supernet v4 — v3_wide + slimmable width + resolution sampling.

Forces arch=v3_wide, slimmable=True. Loads images at 112; before each subnet
forward, adapt_input resizes / gray-replicates per SubnetConfig.

Smoke-test the backbone first::

    python -c "from models.elastic_mbv3 import ElasticFERSuperNet; \
m=ElasticFERSuperNet(8,slimmable=True,arch='v3_wide'); \
c=m.random_config(); import torch; \
y=m(torch.randn(2,3,c.img_size,c.img_size), subnet=c); print(c, y.shape)"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def adapt_input(x: torch.Tensor, cfg: SubnetConfig) -> torch.Tensor:
    if cfg.gray or cfg.img_size == 64:
        g = x.mean(dim=1, keepdim=True)
        x = g.expand(-1, 3, -1, -1)
    if x.shape[-1] != cfg.img_size or x.shape[-2] != cfg.img_size:
        x = F.interpolate(
            x, size=(cfg.img_size, cfg.img_size), mode="bilinear", align_corners=False
        )
    return x


@torch.no_grad()
def eval_subnet(model, loader, device, subnet: SubnetConfig):
    model.eval()
    total = correct = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = adapt_input(x, subnet)
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

    EMA teacher must run *before* any autograd-tracked forward (v3 fix).
    Each path adapts resolution / gray via adapt_input(cfg).
    """
    cfg_max = ElasticFERSuperNet.max_config()

    teacher = None
    if (not max_only) and use_ema_teacher and ema is not None:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            x_t = adapt_input(x, cfg_max)
            teacher = model(x_t, subnet=cfg_max, return_all_exits=False).detach()
        model.load_state_dict(backup, strict=False)
        if was_training:
            model.train()

    x_max = adapt_input(x, cfg_max)
    outs_max = model(x_max, subnet=cfg_max, return_all_exits=True)
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
        x_s = adapt_input(x, cfg)
        outs = model(x_s, subnet=cfg, return_all_exits=True)
        loss = loss + exit_loss(outs, y, crit_weights)
        loss = loss + distill_w * distill_loss(outs[-1], teacher)

    return loss, teacher, outs_max[-1]


def _backup_ckpt(ckpt_path: Path, out_dir: Path) -> None:
    out_env = os.environ.get("OUT")
    if out_env and (Path(out_env) / "backup").is_dir():
        backup_dir = Path(out_env) / "backup"
    else:
        backup_dir = out_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_path, backup_dir / ckpt_path.name)


def main():
    ap = argparse.ArgumentParser(description="Overnight elastic supernet v4 trainer")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, required=True, choices=["ferplus", "expw"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", type=str, default="mild", choices=["none", "mild", "strong"])
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument(
        "--max-path-warmup-epochs",
        type=int,
        default=10,
        help="first N epochs train only max_config (before sandwich)",
    )
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--tag", type=str, default="v4")
    ap.add_argument("--resume", type=Path, default=None, help="full resume (model+opt+sched+epoch)")
    ap.add_argument(
        "--num-exits",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="total exits incl. final; K-ablation uses 1/2/4",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    img_size = 112  # always load full-res; adapt_input downsamples per subnet

    loaders = make_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        aug=args.aug,
        gray_to_3ch=False,
    )
    use_ema = args.ema_decay > 0
    print(
        f"[supernet-v4/{args.dataset_name}] classes={loaders['num_classes']} "
        f"train={loaders['train_len']} img={img_size} bs={args.batch_size} "
        f"arch=v3_wide slimmable=True ema={args.ema_decay if use_ema else 0} "
        f"max_warmup={args.max_path_warmup_epochs} num_exits={args.num_exits} "
        f"device={device}"
    )

    model = ElasticFERSuperNet(
        num_classes=loaders["num_classes"],
        slimmable=True,
        arch="v3_wide",
    ).to(device)
    model.set_num_exits(args.num_exits)
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
    exit_w = None  # exit_loss auto-adapts to len(outs)
    grad_clip = 1.0

    start_epoch = 1
    history = []
    best_max_val = -1.0
    ckpt_path = args.out_dir / f"supernet_{args.dataset_name}_seed{args.seed}{tag}_last.pt"
    best_path = args.out_dir / f"supernet_{args.dataset_name}_seed{args.seed}{tag}_best.pt"
    curve_path = args.out_dir / f"val_curve_{args.dataset_name}_{args.tag or 'v4'}.jsonl"

    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.set_num_exits(args.num_exits)
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
        cfg_max = ElasticFERSuperNet.max_config()  # 112, color
        cfg_min = ElasticFERSuperNet.min_config()  # 64, gray
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            acc_max = eval_subnet(model, loader, device, cfg_max)
            acc_min = eval_subnet(model, loader, device, cfg_min)
            samples = [ElasticFERSuperNet.random_config() for _ in range(4)]
            acc_rand = [eval_subnet(model, loader, device, c) for c in samples]
            model.load_state_dict(backup, strict=False)
        else:
            acc_max = eval_subnet(model, loader, device, cfg_max)
            acc_min = eval_subnet(model, loader, device, cfg_min)
            samples = [ElasticFERSuperNet.random_config() for _ in range(4)]
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
                    use_ema_teacher=use_ema and epoch > args.max_path_warmup_epochs,
                    max_only=epoch <= args.max_path_warmup_epochs,
                )
            if not torch.isfinite(loss).all():
                raise SystemExit("NAN_ABORT")
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            bs = x.size(0)
            loss_val = float(loss.item())
            if not np.isfinite(loss_val):
                raise SystemExit("NAN_ABORT")
            running_loss += loss_val * bs
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
            with curve_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "val_acc_max": acc_max,
                            "val_acc_min": acc_min,
                            "lr": row["lr"],
                        }
                    )
                    + "\n"
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
                _backup_ckpt(best_path, args.out_dir)
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
        _backup_ckpt(ckpt_path, args.out_dir)

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    test_max = eval_subnet(model, loaders["test"], device, ElasticFERSuperNet.max_config())
    test_min = eval_subnet(model, loaders["test"], device, ElasticFERSuperNet.min_config())
    result = {
        "model": "elastic_mbv3_supernet_v4",
        "arch": "v3_wide",
        "slimmable": True,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "tag": args.tag,
        "best_val_acc_max": best_max_val,
        "test_acc_max": test_max,
        "test_acc_min": test_min,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_exits": args.num_exits,
        "enabled_exit_stages": list(model.enabled_exit_stages),
        "img_size": img_size,
        "aug": args.aug,
        "ema_decay": args.ema_decay,
        "max_path_warmup_epochs": args.max_path_warmup_epochs,
        "history": history,
    }
    out_json = args.out_dir / f"supernet_{args.dataset_name}_seed{args.seed}{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
