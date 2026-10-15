#!/usr/bin/env python3
"""Joint C2 + early-exit tracking with the in-band policy swapped for threshold baselines.

C2 subnet selection and the catalog are exactly those of scripts/plot_c2c3_joint_tracking.py;
only the elastic subnet's in-band E[lat](B) curve changes:

  orig         the n=80 probe curve used by the paper figure (c3_randB_l0p05)
  c3_randB     B2 sampled-budget controller on the full FER+ test set, mean over 5 seeds
  thr_fixed    one confidence threshold (t tuned on val to the lambda=0.2 controller), ignores B
  thr_perB     per budget, the val-selected threshold with the highest val accuracy whose
               val latency <= B (earliest threshold if none fits)

Reports MAE |E[lat] - B| over the sweep, OLS slope, and MAE inside the elastic band only.
Outputs: results/stageB/controller/b5_joint_threshold.{csv,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_c2c3_joint_tracking as jt  # noqa: E402

CTRL = ROOT / "results/stageB/controller"
FIXED_T = 0.96


def curves(dev: str, sweep_B: np.ndarray, lam: float) -> dict[str, tuple[list[tuple[float, float]], float]]:
    thr = pd.read_csv(CTRL / "b2_threshold_sweep.csv")
    thr = thr[thr.device == dev].sort_values("t")
    runs = pd.read_csv(CTRL / f"b2_runs_{dev}.csv")
    rb = runs[(runs.kind == "randB") & np.isclose(runs["lambda"], lam)]
    rb = rb.groupby("budget_ms")[["lat_ms", "acc"]].mean().reset_index()

    out = {"c3_randB": (list(zip(rb.budget_ms, rb.lat_ms)), float(rb.acc.mean()))}
    fx = thr.iloc[(thr.t - FIXED_T).abs().argmin()]
    out["thr_fixed"] = ([(float(sweep_B.min()), float(fx.lat_ms)), (float(sweep_B.max()), float(fx.lat_ms))],
                        float(fx.acc))
    pts, accs = [], []
    for B in sweep_B:
        ok = thr[thr.val_lat_ms <= B]
        r = ok.loc[ok.val_acc.idxmax()] if len(ok) else thr.iloc[0]
        pts.append((float(B), float(r.lat_ms)))
        accs.append(float(r.acc))
    out["thr_perB"] = (pts, accs)
    return out


def run_variant(dev: str, curve, acc_by_B) -> pd.DataFrame:
    cat = jt.build_catalog(dev)
    el = next(s for s in cat if s.elastic)
    if curve is not None:
        el.c3_curve = curve
    rows = []
    for B in np.linspace(jt.B_LO, jt.B_HI, jt.N_SWEEP):
        s = jt.select_c2(cat, float(B))
        if s is None:
            continue
        acc = s.test_acc
        if s.elastic and acc_by_B is not None:
            acc = acc_by_B(float(B))
        rows.append({"B": float(B), "lat": s.e_lat(float(B)), "c2_only": s.hi_ms, "elastic": s.elastic,
                     "acc": acc, "band_lo": el.lo_ms, "band_hi": el.hi_ms})
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame) -> dict:
    band = df[df.elastic]
    return {
        "mae_ms": round(float((df.lat - df.B).abs().mean()), 3),
        "mae_c2_only_ms": round(float((df.c2_only - df.B).abs().mean()), 3),
        "ols_slope": round(float(np.polyfit(df.B, df.lat, 1)[0]), 3),
        "n": len(df),
        "n_elastic": len(band),
        "mae_in_band_ms": round(float((band.lat - band.B).abs().mean()), 3) if len(band) else None,
        "overrun_share_in_band": round(float((band.lat > band.B).mean()), 3) if len(band) else None,
        "mean_acc_in_band": round(float(band.acc.mean()), 4) if len(band) else None,
        "in_band_slope": round(float(np.polyfit(band.B, band.lat, 1)[0]), 3) if len(band) > 2 else None,
    }


def plot(df: pd.DataFrame, out: Path) -> None:
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paper_style import H618, T527, TEXT_IN, apply, save

    apply()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.55), sharey=True)
    titles = {"h618": "(a) H618", "t527": "(b) T527"}
    ctrl_color = {"h618": H618, "t527": T527}
    for ax, dev in zip(axes, jt.DEVICES):
        d = df[df.device == dev]
        c3 = d[(d.variant == "c3_randB") & (d.lam == 0.2)]
        tp = d[d.variant == "thr_perB"]
        ax.axvspan(c3.band_lo.iloc[0], min(c3.band_hi.iloc[0], jt.B_HI), color="0.85", alpha=0.35, zorder=0)
        ax.plot([jt.B_LO, jt.B_HI], [jt.B_LO, jt.B_HI], "--", color="0.45", lw=0.8, label="$y = x$")
        ax.plot(c3.B, c3.c2_only, "-", color="0.55", lw=1.2, label="C2 only")
        ax.plot(c3.B, c3.lat, "-", color=ctrl_color[dev], lw=1.4, label="Controller")
        ax.plot(tp.B, tp.lat, ":", color="0.1", lw=1.3, label="Threshold")
        ax.set_xlim(jt.B_LO, jt.B_HI)
        ax.set_ylim(0, jt.B_HI + 1)
        ax.set_xlabel("Budget $B$ (ms)")
        ax.set_title(titles[dev], loc="left", fontsize=8, fontweight="bold", pad=2)
    axes[0].set_ylabel("Achieved latency (ms)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig6_c2c3_joint")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig)


def main() -> int:
    summary, all_rows = {}, []
    sweep_B = np.linspace(jt.B_LO, jt.B_HI, jt.N_SWEEP)
    for lam in (0.05, 0.2):
        for dev in jt.DEVICES:
            cv = curves(dev, sweep_B, lam)
            variants = {"orig": (None, None)}
            for name in ("c3_randB", "thr_fixed"):
                pts, acc = cv[name]
                variants[name] = (pts, lambda B, a=acc: a)
            pts, accs = cv["thr_perB"]
            variants["thr_perB"] = (pts, lambda B, P=pts, A=accs: A[int(np.argmin([abs(p[0] - B) for p in P]))])
            for name, (curve, accf) in variants.items():
                if lam != 0.05 and name in ("orig", "thr_perB", "thr_fixed"):
                    continue
                df = run_variant(dev, curve, accf).assign(device=dev, variant=name, lam=lam)
                all_rows.append(df)
                summary[f"{dev}|{name}|lam{lam}"] = stats(df)
    df = pd.concat(all_rows)
    df.to_csv(CTRL / "b5_joint_threshold.csv", index=False)
    plot(df, CTRL / "b5_joint_threshold")
    (CTRL / "b5_joint_threshold.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
