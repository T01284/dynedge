#!/usr/bin/env python3
"""C3 / G5 — train latency-budget early-exit controller on a frozen supernet.

Design (locked):
  - 2-layer MLP per early exit (exit2/3/4), 8-dim input
  - Gumbel-Softmax τ: 1.0 → 0.3 (linear over epochs)
  - Loss = CE(mixed logits, y) + λ · E[LUT segment latency] / budget_ms
  - 12 configs: {h618, t527, orin} × {5, 10, 20, 50} ms  (MCU excluded)
  - Freeze supernet; train controller only

Examples:
  # single config
  python train/train_controller.py \\
    --ckpt results/supernet/supernet_ferplus_seed42_v4_best.pt \\
    --data-root data/ferplus --dataset-name ferplus \\
    --device h618 --budget-ms 20 --lambda-lat 0.1

  # all 12 configs (default λ=0.1)
  python train/train_controller.py --ckpt ... --data-root ... --dataset-name ferplus --all-configs

  # §8.21 band-aware retrain (board controller_feed + in-band budgets × λ sweep)
  python train/train_controller.py --ckpt ... --data-root data/ferplus --dataset-name ferplus \\
    --band-retrain --lat-feed-dir results \\
    --lambda-lats 0.05,0.1,0.2 --tag v4_band
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lut"))

from datasets.common import make_loaders
from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig
from models.latency_controller import (
    FEAT_DIM,
    LatencyBudgetController,
    expected_latency_ms,
    mix_logits,
)
from topo_ops import expand_subnet_ops, op_key

# H618 fidelity affine (lut/FIDELITY_H618.md) — applied as a*Σsegs + b
H618_AFFINE = {
    "fp32": (0.8786167764789027, 1187.1673261297146),
    "int8": (0.8055378987527299, 877.7509894135619),
}

DEVICES = ("h618", "t527", "orin")
BUDGETS_MS = (5, 10, 20, 50)
DEFAULT_LUT = {
    "h618": ROOT / "lut/ops_h618_v4_full.csv",
    "t527": ROOT / "lut/ops_t527_v4_full.csv",
    "orin": None,  # FLOPs-proxy segments
}


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


def adapt_input(x: torch.Tensor, cfg: SubnetConfig) -> torch.Tensor:
    if cfg.gray or cfg.img_size == 64:
        g = x.mean(dim=1, keepdim=True)
        x = g.expand(-1, 3, -1, -1)
    if x.shape[-1] != cfg.img_size or x.shape[-2] != cfg.img_size:
        x = F.interpolate(
            x, size=(cfg.img_size, cfg.img_size), mode="bilinear", align_corners=False
        )
    return x


def _segment_index(note: str) -> int:
    """Map op note → segment: 0=stem+s0, 1=s1, 2=s2, 3=s3, 4=head."""
    if note.startswith("stem") or "_s0_" in note:
        return 0
    if "_s1_" in note:
        return 1
    if "_s2_" in note:
        return 2
    if "_s3_" in note:
        return 3
    return 4


def load_lut_h618(path: Path) -> dict:
    lut: dict = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok")).lower() not in {"1", "true"}:
                continue
            if r.get("backend") != "h618_ncnn":
                continue
            if r.get("dtype") not in {"fp32", "int8"}:
                continue
            if not r.get("p50_us"):
                continue
            o = dict(r)
            for num in ("cin", "cout", "k", "s", "h", "w", "m", "n", "groups", "width"):
                if o.get(num) in (None, ""):
                    o[num] = None
                else:
                    try:
                        o[num] = float(o[num]) if "." in str(o[num]) else int(float(o[num]))
                    except ValueError:
                        o[num] = None
            o["backend"] = "h618_ncnn"
            lut[op_key(o)] = float(o["p50_us"])
    return lut


def load_lut_t527(path: Path) -> dict:
    lut: dict = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok")).lower() not in {"1", "true"}:
                continue
            if r.get("backend") not in {"ncnn_cpu", "t527_ncnn"}:
                continue
            if r.get("dtype") not in {"fp32", "int8"}:
                continue
            lat = r.get("latency_us_p50") or r.get("p50_us")
            if not lat:
                continue
            o = dict(r)
            for num in ("cin", "cout", "k", "s", "h", "w", "m", "n", "groups", "width"):
                if o.get(num) in (None, ""):
                    o[num] = None
                else:
                    try:
                        o[num] = float(o[num]) if "." in str(o[num]) else int(float(o[num]))
                    except ValueError:
                        o[num] = None
            o["backend"] = "ncnn_cpu"
            lut[op_key(o)] = float(lat)
    return lut


def _lookup_us(lut: dict | None, o: dict) -> float:
    if lut is None:
        # FLOPs-ish proxy from shape
        cin = int(o.get("cin") or o.get("k") or 1)
        cout = int(o.get("cout") or o.get("n") or 1)
        hh = int(o.get("h") or 1)
        ww = int(o.get("w") or hh)
        k = int(o.get("k") or 1)
        return max(0.1, cin * cout * hh * ww * k * k / 1e6)
    v = lut.get(op_key(o))
    if v is None:
        if o.get("op") in {"add", "gap"}:
            cin = int(o.get("cin") or 1)
            hh = int(o.get("h") or 1)
            ww = int(o.get("w") or hh)
            return max(1.0, cin * hh * ww / 1e5)
        return 0.0
    return float(v)


def compute_segment_ms(
    *,
    device_name: str,
    cfg: SubnetConfig,
    dtype: str,
    lut_path: Path | None,
    orin_ref_ms: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return (seg_ms[5], cum_exit_ms[4], meta).

    cum_exit_ms = cumulative cost at exit2, exit3, exit4, final.
    """
    if device_name == "h618":
        backend = "h618_ncnn"
        lut = load_lut_h618(lut_path or DEFAULT_LUT["h618"])
        a, b = H618_AFFINE[dtype]
    elif device_name == "t527":
        backend = "ncnn_cpu"
        lut = load_lut_t527(lut_path or DEFAULT_LUT["t527"])
        a, b = 1.0, 0.0
    elif device_name == "orin":
        backend = "orin_proxy"
        lut = None
        a, b = 1.0, 0.0
    else:
        raise ValueError(f"unknown device {device_name}")

    ops = expand_subnet_ops(
        width=cfg.width,
        depths=cfg.depths,
        img_size=cfg.img_size,
        dtype=dtype if device_name != "orin" else "fp32",
        backend=backend if device_name != "orin" else "h618_ncnn",
    )
    segs = [0.0] * 5
    for o in ops:
        note = str(o.get("note") or "")
        segs[_segment_index(note)] += _lookup_us(lut, o)

    raw = np.array(segs, dtype=np.float64)
    if device_name == "orin":
        # Scale proxy so full-network ≈ orin_ref_ms
        total = raw.sum()
        scale = (orin_ref_ms * 1000.0) / max(total, 1e-6)
        segs_us = raw * scale
        source = f"flops_proxy_scaled_to_{orin_ref_ms}ms"
    else:
        segs_us = a * raw
        # put additive affine overhead on stem segment (always executed)
        segs_us[0] += b
        source = f"lut:{lut_path or DEFAULT_LUT.get(device_name)} a={a} b={b}"

    seg_ms = torch.tensor(segs_us / 1000.0, dtype=torch.float32)
    # cum at exit after stage1/2/3 and final (= all 5 segs)
    cum = torch.stack(
        [
            seg_ms[0] + seg_ms[1],
            seg_ms[0] + seg_ms[1] + seg_ms[2],
            seg_ms[0] + seg_ms[1] + seg_ms[2] + seg_ms[3],
            seg_ms.sum(),
        ]
    )
    meta = {
        "device": device_name,
        "dtype": dtype,
        "seg_ms": seg_ms.tolist(),
        "cum_exit_ms": cum.tolist(),
        "full_ms": float(cum[-1]),
        "source": source,
    }
    return seg_ms, cum, meta


def tau_at_epoch(epoch: int, epochs: int, t0: float, t1: float) -> float:
    if epochs <= 1:
        return t1
    frac = (epoch - 1) / (epochs - 1)
    return t0 + (t1 - t0) * frac


@torch.no_grad()
def evaluate(
    backbone: ElasticFERSuperNet,
    controller: LatencyBudgetController,
    loader,
    device: torch.device,
    subnet: SubnetConfig,
    seg_ms: torch.Tensor,
    cum_ms: torch.Tensor,
    budget_ms: float,
) -> dict:
    backbone.eval()
    controller.eval()
    seg_ms = seg_ms.to(device)
    cum_ms = cum_ms.to(device)
    total = correct = 0
    lat_sum = 0.0
    exit_hist = torch.zeros(4, device=device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = adapt_input(x, subnet)
        with autocast(enabled=device.type == "cuda"):
            e2, e3, e4, ef = backbone(x, subnet=subnet, return_all_exits=True)
            feats = controller.build_features(
                (e2, e3, e4), cum_ms=cum_ms, seg_ms=seg_ms, budget_ms=budget_ms
            )
            gate_logits = controller.forward_gates(feats)
            # hard decisions at eval
            chosen = torch.zeros(x.size(0), dtype=torch.long, device=device)
            all_logits = [e2, e3, e4, ef]
            decided = torch.zeros(x.size(0), dtype=torch.bool, device=device)
            for i, g in enumerate(gate_logits):
                exit_now = g.argmax(dim=-1) == 0
                take = exit_now & ~decided
                chosen = torch.where(take, torch.full_like(chosen, i), chosen)
                decided = decided | take
            chosen = torch.where(decided, chosen, torch.full_like(chosen, 3))
            stacked = torch.stack(all_logits, dim=1)
            logits = stacked[torch.arange(x.size(0), device=device), chosen]
            lat = cum_ms[chosen]
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        lat_sum += lat.sum().item()
        for k in range(4):
            exit_hist[k] += (chosen == k).sum()
    n = max(1, total)
    return {
        "acc": correct / n,
        "lat_ms": lat_sum / n,
        "exit_share": (exit_hist / n).cpu().tolist(),
        "n": total,
    }


def load_controller_feed(path: Path, device_name: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load board-measured seg/cum from exit_segment_latency_v1 summary.json."""
    d = json.loads(path.read_text(encoding="utf-8"))
    feed = d.get("controller_feed") or d.get("cpu") or d
    if feed.get("device") and str(feed["device"]) != device_name:
        raise ValueError(f"feed device={feed.get('device')} != requested {device_name} ({path})")
    seg = torch.tensor([float(x) for x in feed["seg_ms"]], dtype=torch.float32)
    cum = torch.tensor([float(x) for x in feed["cum_exit_ms"]], dtype=torch.float32)
    if seg.numel() != 5 or cum.numel() != 4:
        raise ValueError(f"bad feed shapes seg={tuple(seg.shape)} cum={tuple(cum.shape)} @ {path}")
    # keep non-negative segs (H618 exit4≈final can yield tiny negative Δ)
    seg = torch.clamp(seg, min=0.0)
    full = float(feed.get("full_ms", cum[-1].item()))
    meta = {
        "device": device_name,
        "dtype": feed.get("dtype", "fp32"),
        "seg_ms": seg.tolist(),
        "cum_exit_ms": cum.tolist(),
        "full_ms": full,
        "source": feed.get("source") or str(path),
    }
    return seg, cum, meta


def resolve_lat_feed_path(args, device_name: str) -> Path | None:
    if getattr(args, "lat_feed_json", None):
        return Path(args.lat_feed_json)
    feed_dir = getattr(args, "lat_feed_dir", None)
    if not feed_dir:
        return None
    root = Path(feed_dir)
    cand = [
        root / f"{device_name}_v4" / "exit_segment_latency" / "summary.json",
        root / "exit_segment_latency" / device_name / "summary.json",
        root / f"{device_name}" / "summary.json",
    ]
    for p in cand:
        if p.is_file():
            return p
    raise FileNotFoundError(f"no controller_feed for {device_name} under {root}; tried {cand}")


def band_budgets_from_feed(cum_exit_ms: list[float], n_points: int = 5) -> list[float]:
    """Sample n_points in [earliest_cum, full] inclusive (paper §8.21 option 1)."""
    lo = float(min(cum_exit_ms))
    hi = float(max(cum_exit_ms[-1], max(cum_exit_ms)))
    if hi <= lo:
        return [round(hi, 2)]
    if n_points <= 1:
        return [round(0.5 * (lo + hi), 2)]
    xs = [lo + (hi - lo) * i / (n_points - 1) for i in range(n_points)]
    return [round(x, 2) for x in xs]


def train_one_config(args, device_name: str, budget_ms: float) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    num_classes = 8 if args.dataset_name == "ferplus" else 7
    arch = args.arch or (ckpt.get("args") or {}).get("arch") or "v3_wide"
    backbone = ElasticFERSuperNet(
        num_classes=num_classes, slimmable=True, arch=arch
    )
    backbone.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    subnet = SubnetConfig(
        width=float(args.width),
        depths=parse_depths(args.depths),
        img_size=int(args.img_size),
        gray=bool(args.gray),
    )
    feed_path = resolve_lat_feed_path(args, device_name)
    if feed_path is not None:
        seg_ms, cum_ms, lat_meta = load_controller_feed(feed_path, device_name)
    else:
        lut_path = Path(args.lut) if args.lut else DEFAULT_LUT.get(device_name)
        seg_ms, cum_ms, lat_meta = compute_segment_ms(
            device_name=device_name,
            cfg=subnet,
            dtype=args.dtype,
            lut_path=lut_path,
            orin_ref_ms=args.orin_ref_ms,
        )
    print(
        f"[{device_name} B={budget_ms}ms λ={args.lambda_lat}] full={lat_meta['full_ms']:.2f}ms "
        f"cum={ [round(x,2) for x in lat_meta['cum_exit_ms']] } src={lat_meta['source']}"
    )
    if lat_meta["full_ms"] < budget_ms * 0.5:
        print(
            f"  [warn] full-net {lat_meta['full_ms']:.2f}ms << budget {budget_ms}ms "
            "— controller will mostly take final; still trains for completeness"
        )

    loaders = make_loaders(
        Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=max(112, subnet.img_size),
        aug="mild" if args.dataset_name == "ferplus" else "strong",
        gray_to_3ch=subnet.gray,
    )

    controller = LatencyBudgetController(in_dim=FEAT_DIM, hidden=args.hidden).to(device)
    opt = AdamW(controller.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    lam_tag = f"l{args.lambda_lat:g}".replace(".", "p")
    tag = args.tag or "v4"
    if getattr(args, "random_budget", False):
        out_dir = Path(args.out_dir) / f"{args.dataset_name}_{device_name}_randB_{lam_tag}"
        best_path = (
            out_dir
            / f"controller_{args.dataset_name}_{device_name}_randB_{lam_tag}_seed{args.seed}_{tag}_best.pt"
        )
        pbar_label = f"{device_name}@randB"
    else:
        out_dir = (
            Path(args.out_dir)
            / f"{args.dataset_name}_{device_name}_b{budget_ms:g}ms_{lam_tag}"
        )
        best_path = (
            out_dir
            / f"controller_{args.dataset_name}_{device_name}_b{budget_ms:g}ms_{lam_tag}_seed{args.seed}_{tag}_best.pt"
        )
        pbar_label = f"{device_name}@b{budget_ms}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_d = seg_ms.to(device)
    cum_d = cum_ms.to(device)
    history = []
    best_score = -1e9
    best_val = None

    for epoch in range(1, args.epochs + 1):
        controller.train()
        tau = tau_at_epoch(epoch, args.epochs, args.tau_start, args.tau_end)
        t0 = time.time()
        run_loss = run_ce = run_lat = run_acc = seen = 0.0
        pbar = tqdm(loaders["train"], desc=f"{pbar_label} ep{epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = adapt_input(x, subnet)
            # §8.21 random-B: sample per-batch budget in [0.9·lo, 1.1·hi]
            if getattr(args, "random_budget", False):
                lo = float(getattr(args, "_band_lo", budget_ms))
                hi = float(getattr(args, "_band_hi", budget_ms))
                b_batch = float(torch.empty(1).uniform_(lo, hi).item())
            else:
                b_batch = float(budget_ms)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                with torch.no_grad():
                    e2, e3, e4, ef = backbone(x, subnet=subnet, return_all_exits=True)
                # detach logits so controller only gets CE/latency grads through gates
                exits = (e2.detach(), e3.detach(), e4.detach(), ef.detach())
                feats = controller.build_features(
                    exits[:3], cum_ms=cum_d, seg_ms=seg_d, budget_ms=b_batch
                )
                gate_logits = controller.forward_gates(feats)
                exit_probs, _ = controller.soft_path(gate_logits, tau=tau, hard=False)
                mixed = mix_logits(exit_probs, exits)
                ce = F.cross_entropy(mixed, y)
                elat = expected_latency_ms(exit_probs, cum_d)
                lat_term = (elat / float(b_batch)).mean()
                loss = ce + float(args.lambda_lat) * lat_term
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = x.size(0)
            run_loss += loss.item() * bs
            run_ce += ce.item() * bs
            run_lat += elat.sum().item()
            run_acc += (mixed.argmax(1) == y).float().sum().item()
            seen += bs
            pbar.set_postfix(loss=run_loss / seen, tau=f"{tau:.2f}", B=f"{b_batch:.1f}")

        val = evaluate(
            backbone, controller, loaders["val"], device, subnet, seg_ms, cum_ms, budget_ms
        )
        # score: prefer acc under budget; soft penalty if over
        over = max(0.0, val["lat_ms"] - budget_ms) / budget_ms
        score = val["acc"] - 0.05 * over
        row = {
            "epoch": epoch,
            "tau": tau,
            "train_loss": run_loss / max(1, seen),
            "train_ce": run_ce / max(1, seen),
            "train_lat_ms": run_lat / max(1, seen),
            "train_acc": run_acc / max(1, seen),
            "val_acc": val["acc"],
            "val_lat_ms": val["lat_ms"],
            "val_exit_share": val["exit_share"],
            "score": score,
            "sec": time.time() - t0,
        }
        history.append(row)
        print(
            f"  ep{epoch}: val_acc={val['acc']:.4f} lat={val['lat_ms']:.2f}ms "
            f"exits={[round(x,3) for x in val['exit_share']]} tau={tau:.2f} score={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_val = val
            torch.save(
                {
                    "controller": controller.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "val": val,
                    "lat_meta": lat_meta,
                    "subnet": {
                        "width": subnet.width,
                        "depths": list(subnet.depths),
                        "img_size": subnet.img_size,
                        "gray": subnet.gray,
                    },
                    "device": device_name,
                    "budget_ms": budget_ms,
                    "lambda_lat": args.lambda_lat,
                    "args": vars(args),
                    "feat_dim": FEAT_DIM,
                },
                best_path,
            )

    test = evaluate(
        backbone, controller, loaders["test"], device, subnet, seg_ms, cum_ms, budget_ms
    )
    # reload best for test reporting
    if best_path.is_file():
        blob = torch.load(best_path, map_location=device, weights_only=False)
        controller.load_state_dict(blob["controller"])
        test = evaluate(
            backbone, controller, loaders["test"], device, subnet, seg_ms, cum_ms, budget_ms
        )

    summary = {
        "device": device_name,
        "budget_ms": budget_ms,
        "lambda_lat": float(args.lambda_lat),
        "best_path": str(best_path),
        "best_score": best_score,
        "best_val": best_val,
        "test": test,
        "lat_meta": lat_meta,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{device_name} B={budget_ms}ms] DONE test_acc={test['acc']:.4f} "
        f"lat={test['lat_ms']:.2f}ms -> {best_path}"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, required=True, choices=["ferplus", "expw"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/controller")
    ap.add_argument("--device", type=str, default="h618", choices=list(DEVICES))
    ap.add_argument("--budget-ms", type=float, default=20.0)
    ap.add_argument(
        "--budgets",
        type=str,
        default="",
        help="comma budgets in ms (overrides --budget-ms); e.g. 17,20,23,26,28",
    )
    ap.add_argument(
        "--all-configs",
        action="store_true",
        help="train all 12 Linux configs (3 devices × 4 budgets)",
    )
    ap.add_argument(
        "--band-retrain",
        action="store_true",
        help="§8.21: h618+t527 × in-band budgets from controller_feed (requires --lat-feed-dir)",
    )
    ap.add_argument(
        "--random-budget",
        action="store_true",
        help="sample B~U[0.9·lo,1.1·hi] each batch (board feed band); one controller per device×λ",
    )
    ap.add_argument("--band-points", type=int, default=5, help="points in [lo,hi] for --band-retrain")
    ap.add_argument("--band-lo-scale", type=float, default=0.9, help="random-B lower = scale·lo")
    ap.add_argument("--band-hi-scale", type=float, default=1.1, help="random-B upper = scale·hi")
    ap.add_argument(
        "--lat-feed-json",
        type=Path,
        default=None,
        help="single exit_segment_latency summary.json (for --device)",
    )
    ap.add_argument(
        "--lat-feed-dir",
        type=Path,
        default=None,
        help="dir containing {h618,t527}_v4/exit_segment_latency/summary.json",
    )
    ap.add_argument("--devices", type=str, default="", help="comma devices override (default h618,t527 for band)")
    ap.add_argument("--lut", type=Path, default=None, help="override LUT csv")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--orin-ref-ms", type=float, default=4.0, help="Orin full-net proxy scale (ms)")
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--depths", type=str, default="3,3,3,3")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--arch", type=str, default="v3_wide")
    ap.add_argument("--lambda-lat", type=float, default=0.1, help="λ in CE + λ·E[lat]/B")
    ap.add_argument(
        "--lambda-lats",
        type=str,
        default="",
        help="comma λ sweep; e.g. 0.05,0.1,0.2 (overrides --lambda-lat)",
    )
    ap.add_argument("--tau-start", type=float, default=1.0)
    ap.add_argument("--tau-end", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="v4")
    args = ap.parse_args()

    # Windows may load Linux-pickled PosixPath in ckpt args — only touch model weights.
    if sys.platform == "win32":
        import pathlib

        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]

    lambdas = (
        [float(x) for x in args.lambda_lats.split(",") if x.strip()]
        if args.lambda_lats.strip()
        else [float(args.lambda_lat)]
    )

    device_list: list[str]
    if args.devices.strip():
        device_list = [x.strip() for x in args.devices.split(",") if x.strip()]
    elif args.band_retrain or args.random_budget:
        device_list = ["h618", "t527"]
    elif args.all_configs:
        device_list = list(DEVICES)
    else:
        device_list = [args.device]

    # Build (device, budget) list
    pairs: list[tuple[str, float]] = []
    if args.random_budget:
        if args.lat_feed_dir is None and args.lat_feed_json is None:
            raise SystemExit("--random-budget requires --lat-feed-dir or --lat-feed-json")
        for dev in device_list:
            feed_path = resolve_lat_feed_path(args, dev)
            assert feed_path is not None
            _, _, meta = load_controller_feed(feed_path, dev)
            lo0 = float(min(meta["cum_exit_ms"]))
            hi0 = float(meta["cum_exit_ms"][-1])
            lo = float(args.band_lo_scale) * lo0
            hi = float(args.band_hi_scale) * hi0
            mid = 0.5 * (lo + hi)
            print(f"[randB] {dev} board_band=[{lo0:.2f},{hi0:.2f}] train_U=[{lo:.2f},{hi:.2f}]")
            # stash per-device band on args for train_one_config (set inside loop)
            pairs.append((dev, mid))
        # mark so train_one_config knows
        args.random_budget = True
    elif args.band_retrain:
        if args.lat_feed_dir is None and args.lat_feed_json is None:
            raise SystemExit("--band-retrain requires --lat-feed-dir or --lat-feed-json")
        for dev in device_list:
            # temporarily resolve feed to get band
            feed_path = resolve_lat_feed_path(args, dev)
            assert feed_path is not None
            _, _, meta = load_controller_feed(feed_path, dev)
            if args.budgets.strip():
                buds = [float(x) for x in args.budgets.split(",") if x.strip()]
            else:
                buds = band_budgets_from_feed(meta["cum_exit_ms"], n_points=args.band_points)
            print(f"[band] {dev} cum={meta['cum_exit_ms']} budgets={buds} src={meta['source']}")
            for b in buds:
                pairs.append((dev, float(b)))
    elif args.all_configs:
        pairs = [(d, float(b)) for d in device_list for b in BUDGETS_MS]
    elif args.budgets.strip():
        buds = [float(x) for x in args.budgets.split(",") if x.strip()]
        pairs = [(d, b) for d in device_list for b in buds]
    else:
        pairs = [(device_list[0], float(args.budget_ms))]

    results = []
    for lam in lambdas:
        args.lambda_lat = float(lam)
        for dev, bud in pairs:
            if args.random_budget:
                feed_path = resolve_lat_feed_path(args, dev)
                assert feed_path is not None
                _, _, meta = load_controller_feed(feed_path, dev)
                lo0 = float(min(meta["cum_exit_ms"]))
                hi0 = float(meta["cum_exit_ms"][-1])
                args._band_lo = float(args.band_lo_scale) * lo0
                args._band_hi = float(args.band_hi_scale) * hi0
                print(
                    f"\n======== train controller {dev} randB U[{args._band_lo:.2f},{args._band_hi:.2f}] λ={lam} ========"
                )
            else:
                print(f"\n======== train controller {dev} budget={bud}ms λ={lam} ========")
            results.append(train_one_config(args, dev, bud))

    manifest = {
        "n": len(results),
        "tag": args.tag,
        "lambda_lats": lambdas,
        "configs": [
            {
                "device": r["device"],
                "budget_ms": r["budget_ms"],
                "lambda_lat": r.get("lambda_lat", args.lambda_lat),
                "test_acc": r["test"]["acc"],
                "test_lat_ms": r["test"]["lat_ms"],
                "exit_share": r["test"].get("exit_share"),
                "best_path": r["best_path"],
                "lat_source": (r.get("lat_meta") or {}).get("source"),
            }
            for r in results
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    man_name = f"manifest_{args.dataset_name}_seed{args.seed}"
    if args.band_retrain or args.tag:
        man_name += f"_{args.tag}"
    man_path = Path(args.out_dir) / f"{man_name}.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest {len(results)} configs -> {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
