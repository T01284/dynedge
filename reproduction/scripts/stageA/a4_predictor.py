#!/usr/bin/env python3
"""A4: learned latency predictors vs FLOPs vs the operator LUT (offline, no boards).

Reference latency = affine operator-LUT prediction over the full 972-subnet space
(4 widths x 81 depth tuples x 3 resolutions), per device and dtype, plus AUTO = min over dtype.

1. Subsample curves: train on 10/25/50% of the measured architectures, test on the rest;
   5 random splits. Models: Ridge, small MLP (64-32). Baselines: FLOPs, operator LUT.
2. Few-shot transfer: fit on device A, adapt with k in {0,5,10,20,50} subnets of device B
   (scale+bias calibration and full re-fit), evaluate on the remaining 972-k.
3. Equal-budget search: the actual candidate sets evaluated by the evolutionary search
   (pop 40, 25 generations, ~250 evals) are re-ranked by each proxy and scored against the
   true Pareto front (hypervolume, regret).

Metrics: Kendall tau, MAPE, top-10% hit rate.
Outputs: results/stageA/a4_*.csv|json, fig_a4_subsample.{pdf,png}, fig_a4_fewshot.{pdf,png}
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import (OUT, ROOT, all_subnets, arch_key, features_frame,  # noqa: E402
                    load_op_lut, lut_sum_us, measured_pool, parse_key)

SEARCH_DIR = ROOT / "results/from_autodl/formal_v4"
FRACS = (0.10, 0.25, 0.50)
N_SPLITS = 5
KS = (0, 5, 10, 20, 50)
FEATS = ["width", "img", "gray", "d1", "d2", "d3", "d4", "depth_blocks", "n_layers",
         "mac_stem", "mac_pw", "mac_dw", "mac_se", "mac_head", "mac_fc",
         "mac_stage1", "mac_stage2", "mac_stage3", "mac_stage4", "n_conv", "n_dwconv", "n_gemm", "n_add",
         "flops_search", "params", "act_elems", "dw_share"]
H618_AFFINE = {"fp32": (0.8786167764789027, 1187.1673261297146), "int8": (0.8055378987527299, 877.7509894135619)}


def fit_predict(model, xtr, ytr, xte):
    sc = StandardScaler().fit(xtr)
    ztr, zte = sc.transform(xtr), sc.transform(xte)
    if model == "ridge":
        m = Ridge(alpha=1.0).fit(ztr, np.log(ytr))
    else:
        m = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=800,
                         random_state=0, early_stopping=False, learning_rate_init=1e-3).fit(ztr, np.log(ytr))
    return np.exp(m.predict(zte))


def metrics(y, p, top_frac=0.10):
    y, p = np.asarray(y, float), np.maximum(np.asarray(p, float), 1e-9)
    k = max(1, int(round(top_frac * len(y))))
    hit = len(set(np.argsort(y)[:k]) & set(np.argsort(p)[:k])) / k
    # MAPE after the best log-linear calibration, so scale-free proxies (FLOPs) are comparable.
    lp = np.log(np.clip(p, 1e-6, 1e9))
    A = np.vstack([lp, np.ones(len(p))]).T
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    cal = np.exp(np.clip(coef[0] * lp + coef[1], -20, 20))
    return {"tau": float(kendalltau(y, p).statistic),
            "mape": float(np.mean(np.abs(cal - y) / y)), "topk_hit": float(hit)}


def main() -> int:
    pool = measured_pool()
    subs = all_subnets()
    keys = [arch_key(*s) for s in subs]
    feats = features_frame(keys)
    X = feats[FEATS].to_numpy(float)

    targets, affine = {}, {}
    for dev in ("h618", "t527"):
        lut = load_op_lut(dev)
        sub = pool[pool.device == dev]
        for dt in ("fp32", "int8"):
            a, b = H618_AFFINE[dt] if dev == "h618" else (None, None)
            sums = np.array([lut_sum_us(lut, *s, dt)[0] for s in subs])
            if dev != "h618":
                m = sub[sub.dtype == dt].set_index("key")
                a, b = np.linalg.lstsq(np.vstack([np.array([lut_sum_us(lut, *parse_key(k), dt)[0]
                                                            for k in m.index]), np.ones(len(m))]).T,
                                       m.lat_ms.to_numpy() * 1000, rcond=None)[0]
            affine[(dev, dt)] = (float(a), float(b))
            targets[(dev, dt)] = (a * sums + b) / 1000.0
        targets[(dev, "auto")] = np.minimum(targets[(dev, "fp32")], targets[(dev, "int8")])

    measured_keys = {dev: set(pool[pool.device == dev].key) for dev in ("h618", "t527")}
    idx_meas = {dev: np.array([i for i, k in enumerate(keys) if k in measured_keys[dev]]) for dev in measured_keys}
    for dev in measured_keys:  # sanity: LUT target reproduces measured latency rank
        y = targets[(dev, "fp32")][idx_meas[dev]]
        m = pool[(pool.device == dev) & (pool.dtype == "fp32")].set_index("key").loc[
            [keys[i] for i in idx_meas[dev]]].lat_ms.to_numpy()
        print(f"sanity {dev} fp32 tau(LUT target, measured)={kendalltau(y, m).statistic:.4f} "
              f"MAPE={np.mean(np.abs(y-m)/m):.3f}")

    # ---------------- 1. subsample curves (training pool = the 27 measured architectures)
    rows = []
    for dev in ("h618", "t527"):
        for dt in ("fp32", "int8", "auto"):
            idx = idx_meas[dev]
            y = targets[(dev, dt)][idx]
            for frac in FRACS:
                for split in range(N_SPLITS):
                    rng = np.random.default_rng(1000 * hash((dev, dt, frac)) % 2**31 + split)
                    perm = rng.permutation(len(idx))
                    ntr = max(4, int(round(frac * len(idx))))
                    tr, te = perm[:ntr], perm[ntr:]
                    for model in ("ridge", "mlp"):
                        p = fit_predict(model, X[idx[tr]], y[tr], X[idx[te]])
                        rows.append({"device": dev, "dtype": dt, "frac": frac, "split": split,
                                     "n_train": ntr, "n_test": len(te), "model": model, **metrics(y[te], p)})
                    rows.append({"device": dev, "dtype": dt, "frac": frac, "split": split, "n_train": 0,
                                 "n_test": len(te), "model": "flops", **metrics(y[te], feats.flops_search.to_numpy()[idx[te]])})
                    if dt != "auto":
                        rows.append({"device": dev, "dtype": dt, "frac": frac, "split": split, "n_train": len(idx),
                                     "n_test": len(te), "model": "op_lut",
                                     **metrics(y[te], targets[(dev, dt)][idx[te]])})
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "a4_subsample_raw.csv", index=False)
    agg = sub.groupby(["device", "dtype", "frac", "model"]).agg(
        tau=("tau", "mean"), tau_std=("tau", "std"), mape=("mape", "mean"),
        topk=("topk_hit", "mean")).reset_index()
    agg.to_csv(OUT / "a4_subsample_agg.csv", index=False)

    # ---------------- 2. few-shot transfer
    fs = []
    for src, dst in (("h618", "t527"), ("t527", "h618")):
        for dt in ("fp32", "int8", "auto"):
            ysrc, ydst = targets[(src, dt)], targets[(dst, dt)]
            for model in ("ridge", "mlp"):
                base = fit_predict(model, X, ysrc, X)  # trained on all of source
                for k in KS:
                    for split in range(N_SPLITS):
                        rng = np.random.default_rng(2000 * hash((src, dst, dt, model, k)) % 2**31 + split)
                        pick = rng.permutation(len(keys))
                        cal, te = pick[:k], pick[k:]
                        pred = base.copy()
                        if k >= 2:
                            A = np.vstack([np.log(base[cal]), np.ones(k)]).T
                            coef, *_ = np.linalg.lstsq(A, np.log(ydst[cal]), rcond=None)
                            pred = np.exp(coef[0] * np.log(base) + coef[1])
                        row = {"src": src, "dst": dst, "dtype": dt, "model": model, "adapt": "scale_bias",
                               "k": k, "split": split, **metrics(ydst[te], pred[te])}
                        fs.append(row)
                        if k >= 4:
                            p2 = fit_predict(model, X[cal], ydst[cal], X[te])
                            fs.append({"src": src, "dst": dst, "dtype": dt, "model": model, "adapt": "refit",
                                       "k": k, "split": split, **metrics(ydst[te], p2)})
    fs = pd.DataFrame(fs)
    fs.to_csv(OUT / "a4_fewshot_raw.csv", index=False)
    fsagg = fs.groupby(["src", "dst", "dtype", "model", "adapt", "k"]).agg(
        tau=("tau", "mean"), tau_std=("tau", "std"), mape=("mape", "mean"), topk=("topk_hit", "mean")).reset_index()
    fsagg.to_csv(OUT / "a4_fewshot_agg.csv", index=False)

    # ---------------- 3. equal-budget search re-ranking
    searches = {
        ("ferplus", "h618"): "search_lut/search_ferplus_h618.json",
        ("expw", "h618"): "search_lut/search_expw_h618.json",
        ("expw", "t527"): "search_lut/search_expw_t527.json",
        ("ferplus", "flops"): "search_flops_ctrl/search_ferplus_flops_ctrl.json",
        ("expw", "flops"): "search_flops_ctrl/search_expw_flops_ctrl.json",
    }
    cands = {}
    for (ds, arm), f in searches.items():
        d = json.loads((SEARCH_DIR / f).read_text(encoding="utf-8"))
        cands[(ds, arm)] = [arch_key(c["width"], c["depths"], c["img_size"]) for c in d["candidates"]]
    sr = []
    for ds in ("ferplus", "expw"):
        for dev in ("h618", "t527"):
            y = targets[(dev, "auto")]
            acc = {}
            for arm in ("h618", "t527", "flops"):
                if (ds, arm) not in cands:
                    continue
                f = searches[(ds, arm)]
                d = json.loads((SEARCH_DIR / f).read_text(encoding="utf-8"))
                for c in d["candidates"]:
                    acc.setdefault(arch_key(c["width"], c["depths"], c["img_size"]), c["val_acc"])
            universe = sorted(acc)
            uidx = np.array([keys.index(k) for k in universe])
            ref_lat = y[uidx]
            ref_acc = np.array([acc[k] for k in universe])
            proxies = {"flops": feats.flops_search.to_numpy()[uidx], "op_lut": y[uidx]}
            for model in ("ridge", "mlp"):
                proxies[model] = fit_predict(model, X[idx_meas[dev]], targets[(dev, "auto")][idx_meas[dev]], X[uidx])
            budget = 40  # one generation of the published search (pop 40, 25 generations)
            true_fastest = set(np.argsort(ref_lat)[:budget])
            for name, proxy in proxies.items():
                chosen = set(np.argsort(proxy)[:budget])
                sr.append({"dataset": ds, "device": dev, "proxy": name, "budget": budget,
                           "n_universe": len(universe),
                           "top_overlap": len(chosen & true_fastest) / budget,
                           "best_acc_in_top": float(ref_acc[list(chosen)].max()),
                           "oracle_best_acc": float(ref_acc.max()),
                           "fastest_true_ms": float(ref_lat.min()),
                           "fastest_picked_ms": float(ref_lat[list(chosen)].min())})
    sr = pd.DataFrame(sr)
    sr.to_csv(OUT / "a4_search_rerank.csv", index=False)

    summary = {
        "space": "4 widths x 81 depth tuples x 3 resolutions = 972 subnets",
        "reference": "affine operator-LUT latency (H618 affine = published fit; T527 affine fit on its 27 measured nets)",
        "affine": {f"{d}_{t}": v for (d, t), v in affine.items()},
        "models": "Ridge(alpha=1) and MLP(64,32,relu) on log latency, StandardScaler, 27 features",
        "note_search": "candidate universes are the union of subnets actually evaluated by the evolutionary searches; "
                       "accuracy is the search-time validation accuracy (supernet, not distilled)",
    }
    (OUT / "a4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, dev in zip(axes, ("h618", "t527")):
        for model, c in (("ridge", "C0"), ("mlp", "C1"), ("flops", "0.4")):
            g = agg[(agg.device == dev) & (agg.dtype == "auto") & (agg.model == model)].sort_values("frac")
            ax.errorbar(100 * g.frac, g.tau, yerr=g.tau_std, marker="o", ms=3.5, lw=1, capsize=2, color=c,
                        label={"ridge": "Ridge", "mlp": "MLP (64-32)", "flops": "FLOPs"}[model])
        ax.axhline(1.0, color="C2", ls=":", lw=0.8, label="Operator LUT")
        ax.set_title(dev.upper(), fontsize=8)
        ax.set_xlabel("% of measured subnets used for training", fontsize=6.5)
        ax.set_xticks([10, 25, 50])
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Kendall $\\tau$ on held-out subnets", fontsize=6.5)
    axes[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_a4_subsample.{ext}", dpi=300)
        fig.savefig(ROOT / "paper/Neurocomputing/figures" / f"fig_a4_subsample.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for (src, dst, model), c in ((("h618", "t527", "mlp"), "C1"), (("h618", "t527", "ridge"), "C0"),
                                 (("t527", "h618", "mlp"), "C3")):
        g = fsagg[(fsagg.src == src) & (fsagg.dst == dst) & (fsagg.dtype == "auto") & (fsagg.model == model)
                  & (fsagg.adapt == "scale_bias")].sort_values("k")
        ax.errorbar(g.k, g.tau, yerr=g.tau_std, marker="o", ms=3.5, lw=1, capsize=2, color=c,
                    label=f"{src.upper()}$\\rightarrow${dst.upper()} {model}")
    ax.axhline(1.0, color="C2", ls=":", lw=0.8, label="Operator LUT")
    ax.set_xlabel("Adaptation subnets $k$ on the target device", fontsize=6.5)
    ax.set_ylabel("Kendall $\\tau$ on the other $972-k$", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_a4_fewshot.{ext}", dpi=300)
        fig.savefig(ROOT / "paper/Neurocomputing/figures" / f"fig_a4_fewshot.{ext}", dpi=300)
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(agg[(agg.dtype == "auto")].round(3).to_string(index=False))
    print(fsagg[(fsagg.dtype == "auto") & (fsagg.adapt == "scale_bias")].round(3).to_string(index=False))
    print(sr.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
