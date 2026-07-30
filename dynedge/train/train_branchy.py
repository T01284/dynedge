"""Train BranchyNet-style early-exit ResNet-18 FER baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.branchy_resnet18 import BranchyResNet18
from train.train_common import make_standard_loaders, set_seed, train_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, required=True, choices=["ferplus", "expw"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--aug", type=str, default="mild", choices=["none", "mild", "strong"])
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--exit-weights", type=str, default="0.3,0.3,1.0")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaders = make_standard_loaders(args)
    w2, w3, w4 = [float(x) for x in args.exit_weights.split(",")]
    print(
        f"[branchy/{args.dataset_name}] classes={loaders['num_classes']} "
        f"train={loaders['train_len']} batch={args.batch_size} "
        f"exit_w=({w2},{w3},{w4}) device={device}"
    )
    model = BranchyResNet18(
        num_classes=loaders["num_classes"],
        pretrained=not args.no_pretrained,
    ).to(device)

    def train_step(m, x, y, crit):
        e2, e3, e4 = m(x, return_all=True)
        loss = w2 * crit(e2, y) + w3 * crit(e3, y) + w4 * crit(e4, y)
        return loss, e4

    def eval_logits(m, x):
        return m(x, return_all=False)

    train_loop(
        model=model,
        loaders=loaders,
        device=device,
        args=args,
        model_name="branchy_resnet18",
        train_step_fn=train_step,
        eval_logits_fn=eval_logits,
    )


if __name__ == "__main__":
    main()
