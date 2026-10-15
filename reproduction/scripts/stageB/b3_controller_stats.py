#!/usr/bin/env python3
"""Bootstrap CIs and summary tables for the B1-B4 controller runs.

Latency is the board-measured cumulative exit latency (controller_feed), so a
sample violates budget B when the exit it takes has cum latency > B.
Matched baselines pick the threshold on the validation set, never on test.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageB"))
from b2_controller_exps import FEEDS, band, load_feed, read_csv, write_csv  # noqa: E402

N_BOOT = 2000


def boot_ci(x: np.ndarray, rng) -> tuple[float, float, float]:
    """Mean and 95% percentile CI over test samples (x: (N,) per-sample values)."""
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(m, lo, hi, scale=100.0, nd=1) -> str:
    return f"{m * scale:.{nd}f} [{lo * scale:.{nd}f}, {hi * scale:.{nd}f}]"


class Test:
    def __init__(self, cache: Path):
        b = torch.load(cache / "test.pt", map_location="cpu")
        self.L, self.y = b["logits"].float(), b["y"].long()
        self.pred = self.L.argmax(-1).numpy()  # (N,4)
        self.yn = self.y.numpy()
        v = torch.load(cache / "val.pt", map_location="cpu")
        self.vpred, self.vy = v["logits"].float().argmax(-1).numpy(), v["y"].numpy()

    def correct(self, chosen) -> np.ndarray:
        ch = np.asarray(chosen, dtype=np.int64)
        return (self.pred[np.arange(len(ch)), ch] == self.yn).astype(np.float64)


def matched_threshold(sweep: list[dict], dev: str, val_lat_cap: float) -> str:
    """Threshold with the best val acc among those whose val latency <= cap."""
    cands = [r for r in sweep if r["device"] == dev and float(r["val_lat_ms"]) <= val_lat_cap + 1e-9]
    if not cands:
        cands = [min((r for r in sweep if r["device"] == dev), key=lambda r: float(r["val_lat_ms"]))]
    best = max(cands, key=lambda r: (float(r["val_acc"]), -float(r["val_lat_ms"])))
    return f"{dev}_t{float(best['t']):.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "results/stageB/logits_ferplus_w0.5_d3333_112")
    ap.add_argument("--dir", type=Path, default=ROOT / "results/stageB/controller")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    T = Test(args.cache)
    d = args.dir
    thr_dec = torch.load(d / "b2_threshold_decisions.pt")
    sweep = read_csv(d / "b2_threshold_sweep.csv")
    summary: dict = {}

    # ---- B1: formal checkpoints on the full test set -------------------------
    orig = read_csv(d / "b1_orig_test.csv")
    orig_dec = torch.load(d / "b1_orig_test_decisions.pt")
    b1 = []
    for r in orig:
        dev, B = r["device"], float(r["budget_ms"])
        _, cum = load_feed(dev)
        ch = orig_dec[r["name"]].long().numpy()
        c = T.correct(ch)
        lat = cum.numpy()[ch]
        viol = (lat > B + 1e-6).astype(np.float64)
        ff = T.correct(thr_dec[f"{dev}_fixed3"].long().numpy())
        acc, dacc, v = boot_ci(c, rng), boot_ci(c - ff, rng), boot_ci(viol, rng)
        lat_m = boot_ci(lat, rng)
        b1.append({"device": dev, "kind": r["kind"], "lambda": r["lambda"], "budget_ms": B,
                   "acc": acc[0], "acc_lo": acc[1], "acc_hi": acc[2],
                   "lat_ms": lat_m[0], "lat_lo": lat_m[1], "lat_hi": lat_m[2],
                   "viol": v[0], "viol_lo": v[1], "viol_hi": v[2],
                   "dacc_vs_final": dacc[0], "dacc_lo": dacc[1], "dacc_hi": dacc[2],
                   "final_acc": float(ff.mean()), "final_lat_ms": float(cum[-1])})
    write_csv(d / "b1_table.csv", b1)
    summary["force_final"] = {dev: {"acc": float(T.correct(thr_dec[f"{dev}_fixed3"].long().numpy()).mean()),
                                    "lat_ms": float(load_feed(dev)[1][-1])} for dev in FEEDS}
    summary["exit_acc_test"] = [float((T.pred[:, k] == T.yn).mean()) for k in range(4)]

    # ---- B2/B4: multi-seed runs vs baselines at matched latency -------------
    runs = [r for p in sorted(d.glob("b2_runs_*.csv")) for r in read_csv(p)]
    dec = {}
    for p in sorted(d.glob("b2_test_decisions_*.pt")):
        dec.update(torch.load(p))
    groups = defaultdict(list)
    for r in runs:
        if r["kind"].startswith("randB_drop"):
            continue
        groups[(r["device"], r["kind"], float(r["lambda"]), float(r["budget_ms"]))].append(r)
    b2 = []
    for (dev, kind, lam, B), rs in sorted(groups.items()):
        _, cum = load_feed(dev)
        cumn = cum.numpy()
        c_seeds, thr_lat_seeds, thr_b_seeds, lat_seeds, viol_seeds = [], [], [], [], []
        for r in rs:
            ch = dec[f"{r['name']}@{B:g}"].long().numpy()
            c_seeds.append(T.correct(ch))
            lat_seeds.append(cumn[ch].mean())
            viol_seeds.append((cumn[ch] > B + 1e-6).mean())
            tl = matched_threshold(sweep, dev, float(r["val_lat_ms"]))
            thr_lat_seeds.append(T.correct(thr_dec[tl].long().numpy()))
        tb = matched_threshold(sweep, dev, B)
        cb = T.correct(thr_dec[tb].long().numpy())
        chb = thr_dec[tb].long().numpy()
        c_mean = np.mean(c_seeds, axis=0)
        ff = T.correct(thr_dec[f"{dev}_fixed3"].long().numpy())
        d_lat = boot_ci(c_mean - np.mean(thr_lat_seeds, axis=0), rng)
        d_b = boot_ci(c_mean - cb, rng)
        d_ff = boot_ci(c_mean - ff, rng)
        accs = [c.mean() for c in c_seeds]
        b2.append({"device": dev, "kind": kind, "lambda": lam, "budget_ms": B, "n_seeds": len(rs),
                   "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)) if len(rs) > 1 else 0.0,
                   "lat_mean": float(np.mean(lat_seeds)), "lat_std": float(np.std(lat_seeds, ddof=1)) if len(rs) > 1 else 0.0,
                   "viol_mean": float(np.mean(viol_seeds)), "viol_std": float(np.std(viol_seeds, ddof=1)) if len(rs) > 1 else 0.0,
                   "dacc_vs_thr_latmatched": d_lat[0], "dlm_lo": d_lat[1], "dlm_hi": d_lat[2],
                   "dacc_vs_thr_budget": d_b[0], "dtb_lo": d_b[1], "dtb_hi": d_b[2],
                   "thr_budget": tb, "thr_budget_acc": float(cb.mean()), "thr_budget_lat": float(cumn[chb].mean()),
                   "thr_budget_viol": float((cumn[chb] > B + 1e-6).mean()),
                   "dacc_vs_final": d_ff[0], "dff_lo": d_ff[1], "dff_hi": d_ff[2]})
    write_csv(d / "b2_table.csv", b2)

    # ---- B3: feature ablation of the randB controller -----------------------
    base = defaultdict(list)
    for r in runs:
        if r["kind"] == "randB":
            base[(r["device"], float(r["lambda"]), float(r["budget_ms"]), r["seed"])].append(r)
    b3 = []
    abl = defaultdict(list)
    for r in runs:
        if r["kind"].startswith("randB_drop"):
            abl[(r["device"], float(r["lambda"]), r["drop"])].append(r)
    for (dev, lam, feat), rs in sorted(abl.items()):
        dacc, dviol, dlat = [], [], []
        for r in rs:
            b = base.get((dev, lam, float(r["budget_ms"]), r["seed"]))
            if not b:
                continue
            dacc.append(float(r["acc"]) - float(b[0]["acc"]))
            dviol.append(float(r["viol"]) - float(b[0]["viol"]))
            dlat.append(float(r["lat_ms"]) - float(b[0]["lat_ms"]))
        if dacc:
            b3.append({"device": dev, "lambda": lam, "dropped": feat, "n": len(dacc),
                       "dacc_mean": float(np.mean(dacc)), "dacc_std": float(np.std(dacc, ddof=1)),
                       "dviol_mean": float(np.mean(dviol)), "dviol_std": float(np.std(dviol, ddof=1)),
                       "dlat_mean": float(np.mean(dlat)), "dlat_std": float(np.std(dlat, ddof=1))})
    write_csv(d / "b3_ablation.csv", b3)

    # ---- console digest -------------------------------------------------------
    print("force-final", json.dumps(summary["force_final"]), "exit acc", [round(x, 4) for x in summary["exit_acc_test"]])
    print("\nB1 (formal ckpts, lambda=0.2)")
    for r in b1:
        if float(r["lambda"]) == 0.2:
            print(f"  {r['device']} {r['kind']:5s} B={r['budget_ms']:6.2f} acc {fmt(r['acc'], r['acc_lo'], r['acc_hi'])}"
                  f" lat {r['lat_ms']:.2f} viol {fmt(r['viol'], r['viol_lo'], r['viol_hi'])}"
                  f" dAcc_vs_final {fmt(r['dacc_vs_final'], r['dacc_lo'], r['dacc_hi'], nd=2)}")
    print("\nB2/B4 (5 seeds, lambda=0.2): dAcc vs latency-matched threshold / budget threshold / final (pp)")
    for r in b2:
        if r["lambda"] == 0.2:
            print(f"  {r['device']} {r['kind']:10s} B={r['budget_ms']:6.2f} acc {r['acc_mean'] * 100:.2f}±{r['acc_std'] * 100:.2f}"
                  f" lat {r['lat_mean']:.2f} viol {r['viol_mean'] * 100:.1f}"
                  f" | thrLat {fmt(r['dacc_vs_thr_latmatched'], r['dlm_lo'], r['dlm_hi'], nd=2)}"
                  f" | thrB {fmt(r['dacc_vs_thr_budget'], r['dtb_lo'], r['dtb_hi'], nd=2)} (thr lat {r['thr_budget_lat']:.2f} viol {r['thr_budget_viol'] * 100:.1f})"
                  f" | final {fmt(r['dacc_vs_final'], r['dff_lo'], r['dff_hi'], nd=2)}")
    print("\nB3 ablation (randB, lambda=0.2): mean over seeds x budgets")
    for r in b3:
        if r["lambda"] == 0.2:
            print(f"  {r['device']} -{r['dropped']:16s} dAcc {r['dacc_mean'] * 100:+.2f}±{r['dacc_std'] * 100:.2f}pp"
                  f" dViol {r['dviol_mean'] * 100:+.1f}±{r['dviol_std'] * 100:.1f}pp dLat {r['dlat_mean']:+.2f}ms")
    (d / "b1_b4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
