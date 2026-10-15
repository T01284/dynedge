#!/usr/bin/env python3
"""B1-B4 controller experiments on cached exit logits (b1_cache_logits.py).

Training mirrors train/train_controller.py (per-gate 8->32->2 MLP, Gumbel soft
path, tau 1.0->0.3, CE + lambda*E[lat]/B, AdamW 1e-3, 20 epochs, best-val
checkpoint by acc - 0.05*over). Only the frozen-backbone forward is cached.

Parts:
  orig    B1: re-evaluate the formal checkpoints on the full test set
  train   B2/B3/B4: per-budget, randB, confidence-only and feature-drop runs
Per-sample test decisions are saved so CIs can be bootstrapped later.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.latency_controller import (  # noqa: E402
    LatencyBudgetController,
    expected_latency_ms,
    mix_logits,
)

FEEDS = {
    "h618": ROOT / "results/h618_v4/exit_segment_latency/summary.json",
    "t527": ROOT / "results/t527_v4/exit_segment_latency/summary.json",
}
FEAT_NAMES = ["max_p", "entropy", "rem_over_B", "margin", "spent_over_B", "next_seg_over_B", "exit_idx", "B_over_50"]
BUDGET_FEATS = (2, 4, 5, 7)
LAMBDAS = (0.05, 0.1, 0.2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_feed(dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    f = json.loads(FEEDS[dev].read_text(encoding="utf-8"))["controller_feed"]
    seg = torch.tensor(f["seg_ms"], dtype=torch.float32).clamp(min=0.0)
    cum = torch.tensor(f["cum_exit_ms"], dtype=torch.float32)
    return seg, cum


def band(cum: torch.Tensor, n: int = 5) -> list[float]:
    lo, hi = float(cum.min()), float(max(cum[-1], cum.max()))
    return [round(lo + (hi - lo) * i / (n - 1), 2) for i in range(n)]


def load_cache(d: Path, epochs: int) -> dict:
    def ld(name):
        b = torch.load(d / f"{name}.pt", map_location="cpu")
        return b["logits"].float(), b["y"].long()

    return {
        "train": [ld(f"train_e{e:02d}") for e in range(1, epochs + 1)],
        "val": ld("val"),
        "test": ld("test"),
    }


_FEAT = LatencyBudgetController()


def features(L: torch.Tensor, seg, cum, B: float, drop=()) -> list[torch.Tensor]:
    fs = _FEAT.build_features(L[:, :3].unbind(1), cum_ms=cum, seg_ms=seg, budget_ms=B)
    if drop:
        keep = torch.ones(8)
        keep[list(drop)] = 0.0
        fs = [f * keep for f in fs]
    return fs


@torch.no_grad()
def decide(ctrl, L, seg, cum, B, drop=()) -> torch.Tensor:
    gates = ctrl.forward_gates(features(L, seg, cum, B, drop))
    chosen = torch.full((L.size(0),), 3, dtype=torch.long)
    decided = torch.zeros(L.size(0), dtype=torch.bool)
    for i, g in enumerate(gates):
        take = (g.argmax(-1) == 0) & ~decided
        chosen[take] = i
        decided |= take
    return chosen


def threshold_decide(L: torch.Tensor, t: float) -> torch.Tensor:
    maxp = F.softmax(L[:, :3], dim=-1).amax(-1)  # (N,3)
    chosen = torch.full((L.size(0),), 3, dtype=torch.long)
    decided = torch.zeros(L.size(0), dtype=torch.bool)
    for i in range(3):
        take = (maxp[:, i] >= t) & ~decided
        chosen[take] = i
        decided |= take
    return chosen


def metrics(chosen, L, y, cum, B) -> dict:
    pred = L[torch.arange(L.size(0)), chosen].argmax(-1)
    lat = cum[chosen]
    return {
        "acc": float((pred == y).float().mean()),
        "lat_ms": float(lat.mean()),
        "viol": float((lat > B + 1e-6).float().mean()),
        "exit_share": [float((chosen == k).float().mean()) for k in range(4)],
    }


def train(cache, dev, lam, *, B=None, rand_b=False, drop=(), seed=42, epochs=20, bs=1536,
          tau0=1.0, tau1=0.3):
    set_seed(seed)
    seg, cum = load_feed(dev)
    lo, hi = 0.9 * float(cum.min()), 1.1 * float(cum[-1])
    b_sel = 0.5 * (lo + hi) if rand_b else float(B)
    ctrl = LatencyBudgetController(8, 32)
    opt = AdamW(ctrl.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_score = None, -1e9
    Lv, yv = cache["val"]
    for ep in range(1, epochs + 1):
        tau = tau0 + (tau1 - tau0) * (ep - 1) / max(1, epochs - 1)
        L, y = cache["train"][(ep - 1) % len(cache["train"])]
        perm = torch.randperm(L.size(0))
        ctrl.train()
        for i in range(0, L.size(0) - bs + 1, bs):
            idx = perm[i:i + bs]
            b = float(torch.empty(1).uniform_(lo, hi)) if rand_b else float(B)
            Lb = L[idx]
            probs, _ = ctrl.soft_path(ctrl.forward_gates(features(Lb, seg, cum, b, drop)), tau=tau)
            ce = F.cross_entropy(mix_logits(probs, Lb.unbind(1)), y[idx])
            loss = ce + lam * (expected_latency_ms(probs, cum) / b).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        ctrl.eval()
        m = metrics(decide(ctrl, Lv, seg, cum, b_sel, drop), Lv, yv, cum, b_sel)
        score = m["acc"] - 0.05 * max(0.0, m["lat_ms"] - b_sel) / b_sel
        if score > best_score:
            best_score, best = score, copy.deepcopy(ctrl.state_dict())
    ctrl.load_state_dict(best)
    ctrl.eval()
    return ctrl


def run_orig(cache, out: Path) -> None:
    """B1 on the formal checkpoints: full-test per-sample decisions."""
    import pathlib

    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc]
    Lt, yt = cache["test"]
    rows, dec = [], {}
    for cdir in sorted((ROOT / "results/from_autodl/formal_v4/controller_band").glob("ferplus_*")):
        pts = sorted(cdir.glob("*.pt"), key=lambda q: "retry" not in q.name)
        ref = {}
        if (cdir / "summary.json").is_file():
            s = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))
            want = s["best_path"].replace("\\", "/").rsplit("/", 1)[-1]
            pts = [q for q in pts if q.name == want] or pts
            ref = s["test"]
        if not pts:
            continue
        p = pts[0]
        blob = torch.load(p, map_location="cpu", weights_only=False)
        if blob["device"] not in FEEDS:
            continue
        ctrl = LatencyBudgetController(8, 32)
        ctrl.load_state_dict(blob["controller"])
        ctrl.eval()
        seg = torch.tensor(blob["lat_meta"]["seg_ms"], dtype=torch.float32)
        cum = torch.tensor(blob["lat_meta"]["cum_exit_ms"], dtype=torch.float32)
        rand_b = bool(blob["args"].get("random_budget"))
        budgets = band(cum) if rand_b else [float(blob["budget_ms"])]
        for B in budgets:
            ch = decide(ctrl, Lt, seg, cum, B)
            m = metrics(ch, Lt, yt, cum, B)
            name = f"{blob['device']}_{'randB' if rand_b else 'fixB'}_l{blob['lambda_lat']:g}_B{B:g}"
            dec[name] = ch.to(torch.int8)
            rows.append({"name": name, "device": blob["device"], "kind": "randB" if rand_b else "fixB",
                         "lambda": blob["lambda_lat"], "budget_ms": B, **{k: m[k] for k in ("acc", "lat_ms", "viol")},
                         "exit_share": json.dumps([round(x, 4) for x in m["exit_share"]]),
                         "ckpt": p.name, "ref_acc": ref.get("acc", ""), "ref_lat_ms": ref.get("lat_ms", "")})
            print(name, {k: round(m[k], 4) for k in ("acc", "lat_ms", "viol")}, flush=True)
    write_csv(out / "b1_orig_test.csv", rows)
    torch.save(dec, out / "b1_orig_test_decisions.pt")


def run_train(cache, out: Path, seeds: list[int], parts: set[str], devices: list[str]) -> None:
    Lt, yt = cache["test"]
    Lv, yv = cache["val"]
    suffix = "_".join(devices)
    rows_path = out / f"b2_runs_{suffix}.csv"
    dec_path = out / f"b2_test_decisions_{suffix}.pt"
    done = {r["name"] for r in read_csv(rows_path)}
    dec = torch.load(dec_path) if dec_path.is_file() else {}
    rows = read_csv(rows_path)

    jobs = []
    for dev in devices:
        _, cum = load_feed(dev)
        for seed in seeds:
            for lam in LAMBDAS:
                if "fixB" in parts:
                    for B in band(cum):
                        jobs.append((dev, seed, lam, "fixB", B, ()))
                        jobs.append((dev, seed, lam, "fixB_conf", B, BUDGET_FEATS))
                if "randB" in parts:
                    jobs.append((dev, seed, lam, "randB", None, ()))
                    jobs.append((dev, seed, lam, "randB_conf", None, BUDGET_FEATS))
                if "ablate" in parts:
                    for k in range(8):
                        jobs.append((dev, seed, lam, f"randB_drop{k}", None, (k,)))

    t_start = time.time()
    for n_job, (dev, seed, lam, kind, B, drop) in enumerate(jobs):
        tag = f"{dev}_{kind}_l{lam:g}_s{seed}" + (f"_B{B:g}" if B is not None else "")
        if tag in done:
            continue
        seg, cum = load_feed(dev)
        rand_b = kind.startswith("randB")
        ctrl = train(cache, dev, lam, B=B, rand_b=rand_b, drop=drop, seed=seed)
        for Be in (band(cum) if rand_b else [B]):
            ch = decide(ctrl, Lt, seg, cum, Be, drop)
            m = metrics(ch, Lt, yt, cum, Be)
            mv = metrics(decide(ctrl, Lv, seg, cum, Be, drop), Lv, yv, cum, Be)
            dec[f"{tag}@{Be:g}"] = ch.to(torch.int8)
            rows.append({"name": tag, "device": dev, "kind": kind, "lambda": lam, "seed": seed,
                         "budget_ms": Be, "drop": ",".join(FEAT_NAMES[k] for k in drop),
                         "acc": m["acc"], "lat_ms": m["lat_ms"], "viol": m["viol"],
                         "val_acc": mv["acc"], "val_lat_ms": mv["lat_ms"],
                         "exit_share": json.dumps([round(x, 4) for x in m["exit_share"]])})
        done.add(tag)
        if n_job % 10 == 0 or n_job == len(jobs) - 1:
            write_csv(rows_path, rows)
            torch.save(dec, dec_path)
            print(f"[{n_job + 1}/{len(jobs)}] {tag} {time.time() - t_start:.0f}s", flush=True)
    write_csv(rows_path, rows)
    torch.save(dec, dec_path)


def run_threshold(cache, out: Path) -> None:
    """B2 baseline 1: fixed confidence threshold, swept on val and on test."""
    rows, dec = [], {}
    Lv, yv = cache["val"]
    Lt, yt = cache["test"]
    for dev in FEEDS:
        _, cum = load_feed(dev)
        for t in np.round(np.arange(0.50, 0.9901, 0.01), 2):
            chv, cht = threshold_decide(Lv, float(t)), threshold_decide(Lt, float(t))
            mv = metrics(chv, Lv, yv, cum, float(cum[-1]))
            mt = metrics(cht, Lt, yt, cum, float(cum[-1]))
            dec[f"{dev}_t{t:.2f}"] = cht.to(torch.int8)
            rows.append({"device": dev, "t": float(t), "val_acc": mv["acc"], "val_lat_ms": mv["lat_ms"],
                         "acc": mt["acc"], "lat_ms": mt["lat_ms"],
                         "exit_share": json.dumps([round(x, 4) for x in mt["exit_share"]])})
        for k in range(4):  # fixed exits, k=3 is force-final
            ch = torch.full((Lt.size(0),), k, dtype=torch.long)
            dec[f"{dev}_fixed{k}"] = ch.to(torch.int8)
    write_csv(out / "b2_threshold_sweep.csv", rows)
    torch.save(dec, out / "b2_threshold_decisions.pt")


def read_csv(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "results/stageB/logits_ferplus_w0.5_d3333_112")
    ap.add_argument("--out", type=Path, default=ROOT / "results/stageB/controller")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--parts", default="orig,threshold,fixB,randB,ablate")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--devices", default="h618,t527")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache, args.epochs)
    print("cache", {k: tuple(v[0].shape) for k, v in cache.items() if k != "train"},
          "train epochs", len(cache["train"]), flush=True)
    parts = set(args.parts.split(","))
    if "orig" in parts:
        run_orig(cache, args.out)
    if "threshold" in parts:
        run_threshold(cache, args.out)
    if parts & {"fixB", "randB", "ablate"}:
        run_train(cache, args.out, [int(s) for s in args.seeds.split(",")], parts, args.devices.split(","))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
