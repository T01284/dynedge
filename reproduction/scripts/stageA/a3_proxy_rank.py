#!/usr/bin/env python3
"""A3: per-platform rank fidelity of cost proxies against measured end-to-end latency.

Rows  = device x dtype/backend (measured e2e pool, de-duplicated per architecture).
Cols  = FLOPs (search proxy), MACs (incl. SE/FC), params, #layers, activation elements,
        and the operator-LUT sum where an operator table exists (H618/T527 NCNN).
Stats = Kendall tau-b / Spearman rho with bootstrap 95% CI (resampling subnets),
        discordant-pair rate for FLOPs, list of the worst FLOPs inversions.
Also: full 972-subnet space ranking FLOPs vs LUT-predicted latency (H618/T527).

Outputs: results/stageA/a3_*.csv|json, fig_a3_heatmap.{pdf,png}
"""
from __future__ import annotations

import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

from common import (OUT, PLATFORM_ORDER, ROOT, all_subnets, arch_key, features_frame,  # noqa: E402
                    load_op_lut, lut_sum_us, measured_pool, parse_key)

N_BOOT = 2000
PROXIES = {
    "flops_search": "FLOPs",
    "macs_full": "MACs(+SE)",
    "params": "Params",
    "n_layers": "#Layers",
    "act_elems": "Activations",
    "lut_pred": "Op-LUT",
}
H618_AFFINE = {"fp32": (0.8786167764789027, 1187.1673261297146), "int8": (0.8055378987527299, 877.7509894135619)}
REL_TOL = 0.05


def boot_corr(x, y, fn, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        vals.append(fn(x[i], y[i])[0])
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def affine_fit(s, m):
    A = np.vstack([s, np.ones_like(s)]).T
    a, b = np.linalg.lstsq(A, m, rcond=None)[0]
    return float(a), float(b)


def lut_predictions(pool: pd.DataFrame) -> dict[str, dict[str, float]]:
    """platform -> key -> LUT-predicted ms (affine per dtype; AUTO = min over dtypes)."""
    out: dict[str, dict[str, float]] = {}
    for dev in ("h618", "t527"):
        lut = load_op_lut(dev)
        keys = sorted(pool[pool.device == dev].key.unique())
        sums = {dt: {k: lut_sum_us(lut, *parse_key(k), dt)[0] for k in keys} for dt in ("fp32", "int8")}
        aff = {}
        for dt in ("fp32", "int8"):
            if dev == "h618":
                aff[dt] = H618_AFFINE[dt]
            else:
                sub = pool[(pool.device == dev) & (pool.dtype == dt)]
                aff[dt] = affine_fit(np.array([sums[dt][k] for k in sub.key]), sub.lat_ms.to_numpy() * 1000)
        pred = {dt: {k: (aff[dt][0] * sums[dt][k] + aff[dt][1]) / 1000 for k in keys} for dt in sums}
        out[f"{dev.upper()} NCNN FP32"] = pred["fp32"]
        out[f"{dev.upper()} NCNN INT8"] = pred["int8"]
        out[f"{dev.upper()} NCNN AUTO"] = {k: min(pred["fp32"][k], pred["int8"][k]) for k in keys}
        out[f"_{dev}_affine"] = {dt: aff[dt] for dt in aff}  # type: ignore[assignment]
    return out


def discordant(df: pd.DataFrame, proxy: str = "flops_search"):
    rows, n_pairs, n_disc = [], 0, 0
    for (_, a), (_, b) in itertools.combinations(df.iterrows(), 2):
        if a[proxy] > b[proxy]:
            a, b = b, a
        if b[proxy] < a[proxy] * (1 + REL_TOL):
            continue  # proxy considers them tied
        n_pairs += 1
        if a.lat_ms > b.lat_ms * (1 + REL_TOL):
            n_disc += 1
            rows.append({"lower_proxy_key": a.key, "higher_proxy_key": b.key,
                         "proxy_ratio": b[proxy] / a[proxy], "lat_lower_proxy_ms": a.lat_ms,
                         "lat_higher_proxy_ms": b.lat_ms, "slowdown": a.lat_ms / b.lat_ms})
    return n_pairs, n_disc, sorted(rows, key=lambda r: -r["slowdown"])


def main() -> int:
    pool = measured_pool()
    feats = features_frame(sorted(pool.key.unique()))
    pool = pool.join(feats, on="key")
    lutp = lut_predictions(pool)
    pool["lut_pred"] = [lutp.get(p, {}).get(k, np.nan) for p, k in zip(pool.platform, pool.key)]

    # repeatability of duplicated architectures (same arch, different weights/artifacts)
    rep = pool[(pool.n_rep > 1) & (pool.dtype != "auto")]
    rel_spread = ((rep["max"] - rep["min"]) / rep.lat_ms).to_numpy()

    # dtype as part of the candidate: FP32 and INT8 realisations ranked jointly (FLOPs ties across dtype)
    order = list(PLATFORM_ORDER)
    joint = []
    for dev in ("h618", "t527"):
        u = pool[(pool.device == dev) & (pool.dtype.isin(["fp32", "int8"]))].copy()
        u["platform"] = f"{dev.upper()} NCNN FP32+INT8"
        u["key"] = u.key + "|" + u.dtype
        joint.append(u)
        order.insert(order.index(f"{dev.upper()} NCNN AUTO") + 1, f"{dev.upper()} NCNN FP32+INT8")
    pool_rank = pd.concat([pool] + joint, ignore_index=True)

    rows, worst = [], {}
    for plat in order:
        d = pool_rank[pool_rank.platform == plat].reset_index(drop=True)
        if d.empty:
            continue
        r = {"platform": plat, "n": len(d), "lat_min_ms": d.lat_ms.min(), "lat_max_ms": d.lat_ms.max()}
        for col in PROXIES:
            if d[col].isna().all():
                continue
            t = kendalltau(d[col], d.lat_ms).statistic
            s = spearmanr(d[col], d.lat_ms).statistic
            lo, hi = boot_corr(d[col], d.lat_ms, kendalltau)
            r[f"tau_{col}"], r[f"tau_{col}_lo"], r[f"tau_{col}_hi"] = t, lo, hi
            r[f"rho_{col}"] = s
            r[f"rho_{col}_lo"], r[f"rho_{col}_hi"] = boot_corr(d[col], d.lat_ms, spearmanr)
        n_pairs, n_disc, w = discordant(d)
        r["flops_pairs_compared"], r["flops_pairs_inverted"] = n_pairs, n_disc
        r["flops_inversion_rate"] = n_disc / n_pairs if n_pairs else np.nan
        if w:
            r["worst_inversion"] = (f"{w[0]['lower_proxy_key']} ({w[0]['lat_lower_proxy_ms']:.2f} ms) vs "
                                    f"{w[0]['higher_proxy_key']} ({w[0]['lat_higher_proxy_ms']:.2f} ms), "
                                    f"FLOPs x{w[0]['proxy_ratio']:.2f}, slower x{w[0]['slowdown']:.2f}")
        worst[plat] = w[:5]
        # within-resolution stratified tau for FLOPs (weighted by stratum size)
        taus, ws = [], []
        for _, g in d.groupby("img"):
            if len(g) >= 4 and g.flops_search.nunique() > 1:
                taus.append(kendalltau(g.flops_search, g.lat_ms).statistic)
                ws.append(len(g))
        r["tau_flops_within_res"] = float(np.average(taus, weights=ws)) if taus else np.nan
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "a3_rank_fidelity.csv", index=False)

    # ---------------- full 972-subnet space (LUT-predicted latency as reference)
    full, full_lat = [], {}
    subs = all_subnets()
    fkeys = [arch_key(*s) for s in subs]
    ff = features_frame(fkeys)
    for dev in ("h618", "t527"):
        lut = load_op_lut(dev)
        aff = lutp[f"_{dev}_affine"]
        pr = {}
        for dt in ("fp32", "int8"):
            a, b = aff[dt]
            pr[dt] = np.array([(a * lut_sum_us(lut, *s, dt)[0] + b) / 1000 for s in subs])
        pr["auto"] = np.minimum(pr["fp32"], pr["int8"])
        for dt, v in pr.items():
            t = kendalltau(ff.flops_search.to_numpy(), v).statistic
            tm = kendalltau(ff.macs_full.to_numpy(), v).statistic
            within = np.mean([kendalltau(ff.flops_search.to_numpy()[ff.img.to_numpy() == r],
                                         v[ff.img.to_numpy() == r]).statistic for r in (64, 96, 112)])
            full.append({"device": dev, "dtype": dt, "n": len(v), "tau_flops_vs_lutpred": t,
                         "tau_macs_vs_lutpred": tm, "tau_flops_within_res": within})
        full_lat[dev] = pr
    full = pd.DataFrame(full)
    full.to_csv(OUT / "a3_fullspace_lutpred.csv", index=False)

    # ---------------- inversion rate vs FLOPs gap (near-iso-FLOPs pairs are what a budgeted search compares)
    bins = [1.0, 1.1, 1.25, 1.5, 2.0, np.inf]
    blabels = ["<1.10", "1.10-1.25", "1.25-1.5", "1.5-2", ">2"]

    def inv_by_bin(fl, lat):
        fl, lat = np.asarray(fl, float), np.asarray(lat, float)
        i, j = np.triu_indices(len(fl), 1)
        lo = np.where(fl[i] <= fl[j], i, j)
        hi = np.where(fl[i] <= fl[j], j, i)
        ratio = fl[hi] / fl[lo]
        keep = ratio > 1.0
        inv = lat[lo] > lat[hi]
        out = {}
        for b0, b1, lab in zip(bins[:-1], bins[1:], blabels):
            m = keep & (ratio >= b0) & (ratio < b1)
            out[lab] = (float(inv[m].mean()) if m.any() else np.nan, int(m.sum()))
        return out

    inv_rows = []
    for plat in PLATFORM_ORDER:
        d = pool[pool.platform == plat]
        if len(d) < 15:
            continue
        for lab, (rate, n) in inv_by_bin(d.flops_search, d.lat_ms).items():
            inv_rows.append({"series": f"{plat} (measured)", "flops_gap": lab, "inversion_rate": rate, "n_pairs": n})
    for dev in ("h618", "t527"):
        for dt in ("fp32", "int8"):
            for lab, (rate, n) in inv_by_bin(ff.flops_search, full_lat[dev][dt]).items():
                inv_rows.append({"series": f"{dev.upper()} NCNN {dt.upper()} (972, Op-LUT)", "flops_gap": lab,
                                 "inversion_rate": rate, "n_pairs": n})
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(OUT / "a3_inversion_vs_flops_gap.csv", index=False)

    # dtype decision: how often INT8 beats FP32 on the same architecture
    dt_rows = []
    for dev in ("h618", "t527"):
        piv = pool[pool.device == dev].pivot_table(index="key", columns="dtype", values="lat_ms")
        dt_rows.append({"device": dev, "set": "measured", "n": len(piv),
                        "frac_int8_faster": float((piv["int8"] < piv["fp32"]).mean()),
                        "median_int8_over_fp32": float((piv["int8"] / piv["fp32"]).median())})
        f32, i8 = full_lat[dev]["fp32"], full_lat[dev]["int8"]
        dt_rows.append({"device": dev, "set": "972 Op-LUT", "n": len(f32),
                        "frac_int8_faster": float((i8 < f32).mean()),
                        "median_int8_over_fp32": float(np.median(i8 / f32))})
    dtd = pd.DataFrame(dt_rows)
    dtd.to_csv(OUT / "a3_dtype_decision.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for s, g in inv.groupby("series", sort=False):
        ls = "--" if "Op-LUT" in s else "-"
        ax.plot(range(len(blabels)), g.inversion_rate.to_numpy(), ls, marker="o", ms=2.5, lw=0.9,
                label=s.replace(" (measured)", "").replace(" NCNN", "").replace(" ESP-DL", ""))
    ax.set_xticks(range(len(blabels)))
    ax.set_xticklabels(blabels, fontsize=6)
    ax.set_xlabel("FLOPs ratio between the two subnets of a pair", fontsize=6.5)
    ax.set_ylabel("Rank inversion rate", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=4.6, ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_a3_inversion.{ext}", dpi=300)
        fig.savefig(ROOT / "paper/Neurocomputing/figures" / f"fig_a3_inversion.{ext}", dpi=300)
    plt.close(fig)

    summary = {
        "repeatability_rel_spread_median": float(np.median(rel_spread)) if len(rel_spread) else None,
        "repeatability_rel_spread_p90": float(np.quantile(rel_spread, 0.9)) if len(rel_spread) else None,
        "n_duplicated_arch_rows": int(len(rel_spread)),
        "t527_affine_fit_on_measured": lutp["_t527_affine"],
        "worst_flops_inversions": worst,
    }
    (OUT / "a3_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # ---------------- heatmap
    cols = [c for c in PROXIES]
    mat = np.full((len(tab), len(cols)), np.nan)
    for i, r in tab.iterrows():
        for j, c in enumerate(cols):
            mat[i, j] = r.get(f"tau_{c}", np.nan)
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                        color="k" if mat[i, j] > 0.35 else "w")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([PROXIES[c] for c in cols], rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"{p} (n={n})" for p, n in zip(tab.platform, tab.n)], fontsize=5.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("Kendall $\\tau$ vs. measured latency", fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_a3_heatmap.{ext}", dpi=300)
        fig.savefig(ROOT / "paper/Neurocomputing/figures" / f"fig_a3_heatmap.{ext}", dpi=300)
    plt.close(fig)

    pd.set_option("display.width", 250)
    show = ["platform", "n"] + [f"tau_{c}" for c in cols] + ["tau_flops_search_lo", "tau_flops_search_hi",
                                                            "flops_inversion_rate", "tau_flops_within_res"]
    print(tab[show].round(3).to_string(index=False))
    print(tab[["platform", "worst_inversion"]].to_string(index=False))
    print(full.round(3).to_string(index=False))
    print(inv.pivot_table(index="series", columns="flops_gap", values="inversion_rate", sort=False)
          .reindex(columns=blabels).round(3).to_string())
    print(dtd.round(3).to_string(index=False))
    print({k: summary[k] for k in list(summary)[:4]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
