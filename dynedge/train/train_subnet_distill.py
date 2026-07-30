"""Phase 2.4 — distill fine-tune a fixed-depth subnet inherited from supernet.

Teacher = frozen supernet max path; student = same weights, trained only on target subnet.
"""
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch.nn.functional as F

from datasets.common import make_loaders
from models.elastic_mbv3 import ARCH_SPECS, ElasticFERSuperNet, SubnetConfig, distill_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
def eval_acc(model, loader, device, subnet: SubnetConfig) -> float:
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


def parse_depths(s: str) -> tuple[int, int, int, int]:
    parts = [int(x) for x in s.replace("x", ",").split(",")]
    if len(parts) != 4:
        raise ValueError(f"depths must be 4 ints, got {s}")
    return parts[0], parts[1], parts[2], parts[3]


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
    ap.add_argument("--teacher-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--depths", type=str, required=True, help="e.g. 2,3,3,4")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--arch", type=str, default="", choices=["", "v1", "v3_wide"])
    ap.add_argument(
        "--slimmable",
        action="store_true",
        help="load v4 teacher/student with per-width BN (required for width!=1.0)",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="run val/test every N epochs (last epoch always evaluated); >1 keeps GPU busier",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=112, help="student input size")
    ap.add_argument(
        "--gray",
        action="store_true",
        help="student path: gray→3ch (auto if img-size=64)",
    )
    ap.add_argument("--aug", type=str, default="mild", choices=["none", "mild", "strong"])
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument(
        "--student-act",
        type=str,
        default="hardswish",
        choices=["hardswish", "relu"],
        help="MCU/NPU anchors: train student as native ReLU (no export-time HS→ReLU swap)",
    )
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    depths = parse_depths(args.depths)
    gray = bool(args.gray or args.img_size == 64)
    student_cfg = SubnetConfig(
        width=float(args.width), depths=depths, img_size=args.img_size, gray=gray
    )
    teacher_cfg = ElasticFERSuperNet.max_config()
    dkey = "d" + "".join(str(d) for d in depths)
    wkey = f"w{args.width:g}"
    skey = f"s{args.img_size}{'g' if gray else 'c'}"
    key = f"{wkey}_{dkey}_{skey}"
    tag = f"_{args.tag}" if args.tag else ""

    # Always load 112; adapt_input handles student/teacher resolution.
    loaders = make_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=112,
        aug=args.aug,
    )

    ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    arch = resolve_arch(ckpt, args.arch)
    ckpt_args = ckpt.get("args") or {}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    looks_slimmable = isinstance(state, dict) and any("bns.w" in str(k) for k in state)
    # v4 teachers always use Hardswish + per-width BN banks
    if "v4" in str(args.teacher_ckpt) or "v4" in (args.tag or ""):
        arch = args.arch or "v3_wide"
        slimmable = True
    else:
        slimmable = bool(args.slimmable or ckpt_args.get("slimmable") or looks_slimmable or args.width != 1.0)
    teacher = ElasticFERSuperNet(
        num_classes=loaders["num_classes"], arch=arch, slimmable=slimmable
    ).to(device)
    student = ElasticFERSuperNet(
        num_classes=loaders["num_classes"], arch=arch, slimmable=slimmable
    ).to(device)
    teacher.load_state_dict(state, strict=False)
    student.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    # Teacher stays Hardswish; student may be native ReLU for MCU/NPU deploy anchors.
    if args.student_act == "relu":
        from tools.export_subnet_onnx import hardswish_to_relu

        hardswish_to_relu(student)
        print("[distill] student activation: native ReLU (teacher remains Hardswish)")

    opt = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler(enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    pre_val = eval_acc(student, loaders["val"], device, student_cfg)
    pre_test = eval_acc(student, loaders["test"], device, student_cfg)
    print(
        f"[distill/{args.dataset_name}/{key}] arch={arch} slimmable={slimmable} "
        f"student_act={args.student_act} "
        f"pre_val={pre_val:.4f} pre_test={pre_test:.4f} "
        f"ep={args.epochs} device={device}"
    )

    best_val = -1.0
    history = []
    best_path = args.out_dir / f"subnet_{args.dataset_name}_{key}_seed{args.seed}{tag}_best.pt"
    last_path = args.out_dir / f"subnet_{args.dataset_name}_{key}_seed{args.seed}{tag}_last.pt"

    for epoch in range(1, args.epochs + 1):
        student.train()
        t0 = time.time()
        running, seen = 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"{key} ep{epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_t = adapt_input(x, teacher_cfg)
            x_s = adapt_input(x, student_cfg)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                with torch.no_grad():
                    t_logits = teacher(x_t, subnet=teacher_cfg, return_all_exits=False)
                s_logits = student(x_s, subnet=student_cfg, return_all_exits=False)
                loss = args.ce_weight * ce(s_logits, y) + args.distill_weight * distill_loss(
                    s_logits, t_logits
                )
            if not torch.isfinite(loss).all():
                raise SystemExit("NAN_ABORT")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = x.size(0)
            running += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=running / max(1, seen))

        sched.step()
        do_eval = (epoch % max(1, args.eval_every) == 0) or (epoch == args.epochs)
        if do_eval:
            val = eval_acc(student, loaders["val"], device, student_cfg)
        else:
            val = best_val if best_val >= 0 else pre_val
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_acc": val,
            "evaled": bool(do_eval),
            "lr": opt.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history.append(row)
        print(
            f"{key} epoch {epoch}: loss={row['train_loss']:.4f} val={val:.4f}"
            f"{'' if do_eval else ' (skip-eval)'} "
            f"lr={row['lr']:.5g}"
        )
        payload = {
            "model": student.state_dict(),
            "epoch": epoch,
            "arch": arch,
            "slimmable": slimmable,
            "subnet": {
                "width": student_cfg.width,
                "depths": list(depths),
                "img_size": student_cfg.img_size,
                "gray": student_cfg.gray,
                "act": args.student_act,
            },
            "student_act": args.student_act,
            "best_val": best_val,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        torch.save(payload, last_path)
        if do_eval and val > best_val:
            best_val = val
            payload["best_val"] = best_val
            torch.save(payload, best_path)

    ckpt_best = torch.load(best_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt_best["model"])
    test = eval_acc(student, loaders["test"], device, student_cfg)
    result = {
        "model": "elastic_mbv3_subnet_distill",
        "arch": arch,
        "slimmable": slimmable,
        "dataset": args.dataset_name,
        "key": key,
        "width": student_cfg.width,
        "depths": list(depths),
        "img_size": student_cfg.img_size,
        "gray": student_cfg.gray,
        "student_act": args.student_act,
        "seed": args.seed,
        "tag": args.tag,
        "pre_val": pre_val,
        "pre_test": pre_test,
        "best_val": best_val,
        "test_acc": test,
        "epochs": args.epochs,
        "teacher_ckpt": str(args.teacher_ckpt),
        "history": history,
        "note": (
            "formal distill; MCU/NPU anchors use --student-act relu "
            "(native ReLU, not export-time HS→ReLU)"
        ),
    }
    out_json = args.out_dir / f"subnet_{args.dataset_name}_{key}_seed{args.seed}{tag}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
