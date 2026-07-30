"""Step-A: train a standalone fixed-depth MBV3-FER from scratch (supernet-tax diagnosis)."""
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
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.common import make_loaders
from models.standalone_mbv3 import StandaloneMBV3FER


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


def parse_depths(s: str) -> tuple[int, int, int, int]:
    parts = [int(x) for x in s.replace("x", ",").split(",")]
    if len(parts) != 4:
        raise ValueError(f"depths must be 4 ints, got {s}")
    return parts[0], parts[1], parts[2], parts[3]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(x)
            loss = crit(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return {"acc": correct / max(1, total), "loss": loss_sum / max(1, total)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, required=True, choices=["ferplus", "expw"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--depths", type=str, required=True, help="e.g. 4,4,4,4 or 2,2,2,2")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--aug", type=str, default="mild", choices=["none", "mild", "strong"])
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--arch", type=str, default="v1", choices=["v1", "v3_wide"])
    ap.add_argument(
        "--act",
        type=str,
        default="hardswish",
        choices=["hardswish", "relu"],
        help="Job0: swap Hardswish→ReLU for deploy-cost probe (else keep Hardswish)",
    )
    ap.add_argument(
        "--no-se",
        action="store_true",
        help="Disable SE on v3_wide (Identity); Job0-clone SE ablation",
    )
    ap.add_argument("--tag", type=str, default="standalone")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    depths = parse_depths(args.depths)
    key = "d" + "".join(str(d) for d in depths)
    tag = f"_{args.tag}" if args.tag else ""
    arch_tag = args.arch

    loaders = make_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        aug=args.aug,
    )
    model = StandaloneMBV3FER(
        num_classes=loaders["num_classes"],
        depths=depths,
        arch=args.arch,
        use_se=False if args.no_se else None,
    ).to(device)
    if args.act == "relu":
        from tools.export_subnet_onnx import hardswish_to_relu

        hardswish_to_relu(model)
        print("[standalone] activation swapped: Hardswish → ReLU")
    if args.no_se:
        print("[standalone] SE disabled (use_se=False)")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_ema = args.ema_decay > 0
    print(
        f"[standalone/{args.dataset_name}/{key}/{args.arch}] params={n_params/1e6:.2f}M "
        f"img={args.img_size} act={args.act} se={not args.no_se} ep={args.epochs} "
        f"aug={args.aug} ema={args.ema_decay if use_ema else 0} device={device}"
    )

    if args.optimizer == "sgd":
        opt = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
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
    ema = ModelEMA(model, args.ema_decay) if use_ema else None
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val = -1.0
    history = []
    best_path = (
        args.out_dir
        / f"standalone_{args.dataset_name}_{key}_{arch_tag}_seed{args.seed}{tag}_best.pt"
    )
    last_path = (
        args.out_dir
        / f"standalone_{args.dataset_name}_{key}_{arch_tag}_seed{args.seed}{tag}_last.pt"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, running_acc, seen = 0.0, 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"{key} ep{epoch}/{args.epochs}", leave=False)
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
            running_acc += (logits.argmax(1) == y).float().sum().item()
            seen += bs
            pbar.set_postfix(loss=running_loss / max(1, seen), acc=running_acc / max(1, seen))

        sched.step()

        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            val = evaluate(model, loaders["val"], device)
            model.load_state_dict(backup, strict=False)
        else:
            val = evaluate(model, loaders["val"], device)

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "train_acc": running_acc / max(1, seen),
            "val_acc": val["acc"],
            "val_loss": val["loss"],
            "lr": opt.param_groups[0]["lr"],
            "sec": time.time() - t0,
            "imgs_per_sec": seen / max(1e-6, time.time() - t0),
        }
        history.append(row)
        print(
            f"{key} epoch {epoch}: train_acc={row['train_acc']:.4f} "
            f"val_acc={val['acc']:.4f} lr={row['lr']:.5g} {row['imgs_per_sec']:.0f} img/s"
        )

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ema is not None:
            for k, v in ema.state.items():
                if k in state:
                    state[k] = v.detach().cpu().clone()

        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state if ema is not None else None,
                "epoch": epoch,
                "best_val": best_val,
                "depths": list(depths),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "history": history,
            },
            last_path,
        )
        if val["acc"] > best_val:
            best_val = val["acc"]
            torch.save(
                {
                    "model": state,
                    "epoch": epoch,
                    "best_val": best_val,
                    "depths": list(depths),
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "ema_decay": args.ema_decay,
                    "params": n_params,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, loaders["test"], device)
    if args.no_se:
        experiment = (
            "hs_sefree_ablation" if args.act == "hardswish" else "job0_sefree_ablation"
        )
        pass_rule = "SE-free ablation: fill act×SE 2x2; informational only"
    elif args.act == "relu":
        experiment = "job0_relu_cost"
        pass_rule = "Job0: report Δ vs Hardswish Step-B; does not gate v4"
    elif args.arch != "v1":
        experiment = "step_b_capacity_boost"
        pass_rule = "Step-B capacity gate"
    else:
        experiment = "step_a_supernet_tax"
        pass_rule = "Step-A supernet-tax diagnosis"
    result = {
        "model": "standalone_mbv3_fer",
        "arch": args.arch,
        "act": args.act,
        "use_se": not args.no_se,
        "experiment": experiment,
        "dataset": args.dataset_name,
        "key": key,
        "depths": list(depths),
        "seed": args.seed,
        "tag": args.tag,
        "params": n_params,
        "best_val_acc": best_val,
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "aug": args.aug,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "img_size": args.img_size,
        "history": history,
        "compare_note": {
            "step_b_v3_wide_d4444_img112_hardswish_test": 0.8728,
            "job0_relu_se_test": 0.8617,
            "resnet18_k2_mean_test": 0.8849,
            "pass_rule": pass_rule,
        },
    }
    out_json = (
        args.out_dir
        / f"standalone_{args.dataset_name}_{key}_{arch_tag}_seed{args.seed}{tag}.json"
    )
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
