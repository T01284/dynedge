#!/usr/bin/env python3
"""A9: feasibility of a dtype-aware search claim (route B).

Claim under test: "FLOPs search + default INT8 deployment" vs "measured LUT + joint dtype
decision", compared at iso-accuracy on measured board latency.

1. Per-architecture INT8/FP32 latency ratio on the measured NCNN pool (27 archs per board),
   and FP16 vs INT8 on the Orin TensorRT pool.
2. Iso-accuracy latency ratio on the C1 points (same nearest-neighbour pairing as A1), for
   several baselines, so the headline ratio can be split into a dtype part and a search part.
3. Regret of a per-device dtype rule (always FP32) against the per-architecture oracle.
4. How well the operator LUT predicts the per-architecture dtype decision.
5. Board agreement of INT8 with FP32 (the C1 accuracy axis uses GPU accuracy for both dtypes).

Outputs: results/stageA/a9_*.csv, a9_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from common import OUT, ROOT, bootstrap_ci, load_op_lut, lut_sum_us, measured_pool, parse_key

N_BOOT = 10000


def ratio_stats(x, label: str) -> dict:
    x = np.asarray(x, dtype=float)
    lx = np.log(x)
    lo, hi = bootstrap_ci(lx, n_boot=N_BOOT)
    nz = lx[lx != 0]
    return {
        "condition": label, "n": int(len(x)), "median": float(np.median(x)),
        "geomean": float(np.exp(lx.mean())), "ci_lo": float(np.exp(lo)), "ci_hi": float(np.exp(hi)),
        "min": float(x.min()), "max": float(x.max()), "frac_gt1": float((x > 1).mean()),
        "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 1 else float("nan"),
        "sign_p": float(binomtest(int((x > 1).sum()), int((x != 1).sum())).pvalue) if (x != 1).any() else float("nan"),
    }


def per_arch_dtype() -> tuple[pd.DataFrame, list[dict]]:
    pool = measured_pool()
    rows, stats = [], []
    for dev, (a, b) in {"h618": ("int8", "fp32"), "t527": ("int8", "fp32"), "orin": ("int8", "fp16")}.items():
        sub = pool[(pool.device == dev) & pool.dtype.isin([a, b])]
        piv = sub.pivot_table(index="key", columns="dtype", values="lat_ms").dropna()
        piv["ratio"] = piv[a] / piv[b]
        for k, r in piv.iterrows():
            rows.append({"device": dev, "key": k, f"lat_{b}": r[b], "lat_int8": r[a], "int8_over_other": r["ratio"]})
        stats.append(ratio_stats(piv["ratio"], f"{dev.upper()} INT8/{b.upper()} per arch"))
    return pd.DataFrame(rows), stats


def iso_acc_policies() -> tuple[pd.DataFrame, list[dict]]:
    p = pd.read_csv(ROOT / "results/c1_flops_vs_lut/points.csv").drop(columns=["key"])
    p = p.rename(columns={"short": "key", "latency_ms_p50": "lat", "plot_acc": "acc"})
    lat = p.pivot_table(index=["arm", "device", "dataset", "key"], columns="dtype", values="lat").reset_index()
    acc = p.groupby(["arm", "device", "dataset", "key"]).acc.first().reset_index()
    pts = lat.merge(acc)
    pts["auto"] = pts[["fp32", "int8"]].min(axis=1)
    rows = []
    for (dev, ds), g in pts.groupby(["device", "dataset"]):
        L, F = g[g.arm == "lut"], g[g.arm == "flops"]
        for _, l in L.iterrows():
            f = F.iloc[(F.acc - l.acc).abs().to_numpy().argmin()]
            rows.append({
                "device": dev, "dataset": ds, "lut_key": l.key, "flops_key": f.key,
                "lut_acc": l.acc, "flops_acc": f.acc,
                # headline route-B comparison and its decomposition
                "flopsINT8_over_lutAUTO": f.int8 / l.auto,
                "flopsFP32_over_lutAUTO": f.fp32 / l.auto,
                "flopsAUTO_over_lutAUTO": f.auto / l.auto,
                "flopsINT8_over_flopsFP32": f.int8 / f.fp32,
                "flopsINT8_over_lutINT8": f.int8 / l.int8,
                "lutINT8_over_lutAUTO": l.int8 / l.auto,
                "lut_auto_is_int8": bool(l.int8 < l.fp32),
            })
    df = pd.DataFrame(rows)
    labels = {
        "flopsINT8_over_lutAUTO": "Route B as stated: FLOPs+INT8 vs LUT+dtype",
        "flopsINT8_over_flopsFP32": "  dtype only: same FLOPs subnets, INT8 vs FP32",
        "flopsFP32_over_lutAUTO": "  fair baseline: FLOPs+FP32 (per-device rule) vs LUT+dtype",
        "flopsAUTO_over_lutAUTO": "  search only: both arms per-arch dtype (= A1 AUTO)",
        "flopsINT8_over_lutINT8": "  paper Table III: both arms INT8 (= A1 INT8)",
    }
    stats = [ratio_stats(df[c], lab) for c, lab in labels.items()]
    # cluster bootstrap over LUT subnets for the headline, as in A1
    df["cluster"] = df.dataset + "|" + df.lut_key
    rng = np.random.default_rng(0)
    for c in ("flopsINT8_over_lutAUTO", "flopsFP32_over_lutAUTO"):
        groups = [np.log(g[c].to_numpy()) for _, g in df.groupby("cluster")]
        meds = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
                for _ in range(N_BOOT)]
        lo, hi = np.quantile(meds, [0.025, 0.975])
        stats.append({"condition": f"{labels[c].strip()} (cluster bootstrap, 17 LUT subnets)",
                      "n": len(df), "median": float(np.median(df[c])), "ci_lo": float(np.exp(lo)),
                      "ci_hi": float(np.exp(hi))})
    return df, stats


def device_rule_regret(arch: pd.DataFrame) -> list[dict]:
    out = []
    for dev in ("h618", "t527"):
        a = arch[arch.device == dev]
        auto = np.minimum(a.lat_fp32, a.lat_int8)
        regret = a.lat_fp32 / auto
        # probability that a single random architecture, benched in both dtypes, picks FP32
        p_pick_fp32 = float((a.int8_over_other > 1).mean())
        out.append({
            "device": dev, "n": len(a), "n_int8_faster": int((a.int8_over_other < 1).sum()),
            "always_fp32_regret_median": float(np.median(regret)), "always_fp32_regret_max": float(regret.max()),
            "always_fp32_regret_geomean": float(np.exp(np.log(regret).mean())),
            "always_int8_penalty_geomean": float(np.exp(np.log(a.lat_int8 / auto).mean())),
            "p_one_arch_bench_picks_fp32": p_pick_fp32,
            "int8_faster_keys": a[a.int8_over_other < 1].key.tolist(),
            "int8_faster_gain": (1 / a[a.int8_over_other < 1].int8_over_other).round(3).tolist(),
        })
    return out


def lut_dtype_decision(arch: pd.DataFrame) -> list[dict]:
    out = []
    for dev in ("h618", "t527"):
        lut = load_op_lut(dev)
        a = arch[arch.device == dev]
        hit, pred_ratio = 0, []
        for _, r in a.iterrows():
            w, d, img = parse_key(r.key)
            f32, _ = lut_sum_us(lut, w, d, img, "fp32")
            i8, _ = lut_sum_us(lut, w, d, img, "int8")
            pred_ratio.append(i8 / f32)
            hit += int((i8 < f32) == (r.int8_over_other < 1))
        out.append({"device": dev, "n": len(a), "lut_decision_agrees": hit,
                    "lut_pred_int8_over_fp32_median": float(np.median(pred_ratio)),
                    "measured_int8_over_fp32_median": float(np.median(a.int8_over_other))})
    return out


def int8_agreement() -> list[dict]:
    p = pd.read_csv(ROOT / "results/c1_flops_vs_lut/points.csv")
    p = p[(p.dtype == "int8") & (p.device == "h618")].dropna(subset=["board_int8_acc"])
    out = []
    for ds, g in p.groupby("dataset"):
        g = g.drop_duplicates("short")
        d = 100 * (g.board_int8_acc - g.board_fp32_acc)
        out.append({"dataset": ds, "n_subnets": len(g),
                    "int8_minus_fp32_pp_mean": float(d.mean()), "int8_minus_fp32_pp_min": float(d.min()),
                    "int8_minus_fp32_pp_max": float(d.max()),
                    "fp32_int8_agree_mean": float(g.board_fp32_int8_agree.mean()),
                    "fp32_int8_agree_min": float(g.board_fp32_int8_agree.min())})
    return out


def main() -> int:
    arch, s1 = per_arch_dtype()
    arch.to_csv(OUT / "a9_per_arch_dtype.csv", index=False)
    pairs, s2 = iso_acc_policies()
    pairs.to_csv(OUT / "a9_iso_acc_policies.csv", index=False)
    table = pd.DataFrame(s1 + s2)
    table.to_csv(OUT / "a9_table.csv", index=False)
    summary = {
        "per_arch_and_policies": table.to_dict(orient="records"),
        "device_rule_regret": device_rule_regret(arch),
        "lut_dtype_decision": lut_dtype_decision(arch),
        "int8_vs_fp32_board_agreement": int8_agreement(),
        "lut_arm_auto_picks_int8": int(pairs.lut_auto_is_int8.sum()),
        "n_pairs": len(pairs),
    }
    (OUT / "a9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(table[["condition", "n", "median", "ci_lo", "ci_hi", "frac_gt1", "wilcoxon_p"]].to_string(index=False))
    for k in ("device_rule_regret", "lut_dtype_decision", "int8_vs_fp32_board_agreement"):
        print(k, json.dumps(summary[k], indent=1))
    print("LUT arm per-arch auto picks INT8:", summary["lut_arm_auto_picks_int8"], "/", len(pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
