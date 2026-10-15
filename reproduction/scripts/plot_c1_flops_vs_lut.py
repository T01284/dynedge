#!/usr/bin/env python3
"""B-1/G-1: C1 FLOPs vs LUT dual Pareto frontiers + iso tables (H618/T527).

Uses:
  - FLOPs board: results/{h618,t527}_v4_flops_ctrl_baked/matrix_summary.csv
  - LUT board:   results/{h618,t527}_v4_formal_baked(+_expw)
  - B12 latency override: the 10 ReLU 64g subnets were benched as Hardswish graphs;
    their P50 comes from the rebuilt graphs (results/stageB/c1_fix/latency_compare.csv).
    Hardswish subnets keep the original values (rebench within +-1.2%).
  - Acc (frontier Y): GPU distill test_acc from formal_v4/*.json
  - Acc (deploy fidelity): board NCNN FP32/INT8 top-1 on the full test split (B11 --fixed)

Outputs under results/c1_flops_vs_lut/:
  points.csv, iso_acc.csv, iso_lat.csv,
  frontier_{device}_{dataset}_int8.png, SUMMARY snippet text.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import H618, T527, TEXT_IN, apply, save
OUT = ROOT / "results" / "c1_flops_vs_lut"
FIG = ROOT / "paper" / "Neurocomputing" / "figures"
CKPT_FLOPS = ROOT / "results/from_autodl/formal_v4/subnets_v4_flops_ctrl"
CKPT_LUT = ROOT / "results/from_autodl/formal_v4/subnets_v4_formal"

LAT_FIX_CSV = ROOT / "results/stageB/c1_fix/latency_compare.csv"
BOARD_ACC_SUMMARIES = [
    ROOT / "results/stageB/board_full_acc/fixed/summary.csv",
    ROOT / "results/stageB/board_full_acc/expw/fixed/summary.csv",
]


def parse_stem_meta(model: str) -> dict:
    m = re.search(
        r"subnet_(ferplus|expw)_w([0-9.]+)_d(\d{4})_s(\d+)([gc])_", model
    )
    if not m:
        return {}
    ds, w, d, img, gc = m.groups()
    return {
        "dataset": ds,
        "width": float(w),
        "depths": d,
        "img": int(img),
        "gray": gc == "g",
        "short": f"w{w}_d{d}_s{img}{gc}",
    }


def key_of_model(model: str) -> str | None:
    meta = parse_stem_meta(model)
    if not meta:
        return None
    return f"{meta['dataset']}|{meta['short']}"


def load_acc(ckpt_dir: Path, tag: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in ckpt_dir.glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        name = p.name.replace("_best", "").replace(".json", "")
        name = name.replace(f"_seed42_{tag}", "")
        m = re.search(r"subnet_(ferplus|expw)_(w[\d.]+_d\d+_s\d+[gc])", name)
        if not m:
            continue
        # skip multiseed extras for seed42 primary map
        if "_s1234_" in p.name or "_s2026_" in p.name:
            continue
        key = f"{m.group(1)}|{m.group(2)}"
        acc = d.get("test_acc")
        if acc is not None:
            out[key] = float(acc)
    return out


def load_relu_latency() -> dict[tuple[str, str, str], float]:
    """(device, model, dtype) -> rebuilt-graph P50 (us) for the ReLU-rebaked subnets."""
    out: dict[tuple[str, str, str], float] = {}
    with LAT_FIX_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["relu_rebake"] == "True" and r["latency_us_p50_new"]:
                out[(r["board"], r["model"], r["dtype"])] = float(r["latency_us_p50_new"])
    return out


def load_board_acc() -> dict[tuple[str, str], dict]:
    """(board, model) -> {board_fp32_acc, board_int8_acc, board_fp32_int8_agree}."""
    out: dict[tuple[str, str], dict] = {}
    for path in BOARD_ACC_SUMMARIES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out[(r["board"], r["model"])] = {
                    "board_fp32_acc": float(r["fp32_acc"]),
                    "board_int8_acc": float(r["int8_acc"]),
                    "board_fp32_int8_agree": float(r["fp32_int8_agree"]),
                }
    return out


def load_matrix(path: Path, arm: str, device: str) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            model = r.get("model") or r.get("stem") or ""
            dtype = (r.get("dtype") or "").lower()
            if dtype not in {"fp32", "int8"}:
                continue
            p50 = None
            for k in ("latency_us_p50", "p50_us"):
                if r.get(k):
                    p50 = float(r[k])
                    break
            if p50 is None and r.get("latency_ms_p50"):
                p50 = float(r["latency_ms_p50"]) * 1000.0
            if p50 is None and r.get("p50_ms"):
                p50 = float(r["p50_ms"]) * 1000.0
            if p50 is None:
                continue
            meta = parse_stem_meta(model)
            if not meta:
                continue
            rows.append(
                {
                    "arm": arm,
                    "device": device,
                    "dataset": meta["dataset"],
                    "dtype": dtype,
                    "model": model,
                    "short": meta["short"],
                    "key": f"{meta['dataset']}|{meta['short']}",
                    "latency_ms_p50": p50 / 1000.0,
                    "width": meta["width"],
                    "depths": meta["depths"],
                    "img": meta["img"],
                    "gray": meta["gray"],
                }
            )
    return rows


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(points, key=lambda x: (x[0], -x[1]))
    front: list[tuple[float, float]] = []
    best_acc = -1.0
    for lat, acc in pts:
        if acc > best_acc + 1e-12:
            front.append((lat, acc))
            best_acc = acc
    return front


def nearest(target: float, items: list[tuple[float, dict]], key_idx: int = 0):
    return min(items, key=lambda x: abs(x[key_idx] - target))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    acc_f = load_acc(CKPT_FLOPS, "v4_flops_ctrl")
    acc_l = load_acc(CKPT_LUT, "v4_formal_lut")

    relu_lat = load_relu_latency()
    board_acc = load_board_acc()
    # H618 and T527 run the same fer_ncnn_acc build and agree image-by-image (B11),
    # so a model measured on one board stands in for the other.
    board_acc_any = {m: v for (_, m), v in board_acc.items()}

    mats = []
    mats += load_matrix(ROOT / "results/h618_v4_flops_ctrl_baked/matrix_summary.csv", "flops", "h618")
    mats += load_matrix(ROOT / "results/t527_v4_flops_ctrl_baked/matrix_summary.csv", "flops", "t527")
    mats += load_matrix(ROOT / "results/h618_v4_formal_baked/matrix_summary.csv", "lut", "h618")
    mats += load_matrix(ROOT / "results/t527_v4_formal_baked/matrix_summary.csv", "lut", "t527")
    mats += load_matrix(ROOT / "results/h618_v4_formal_baked_expw/matrix_summary.csv", "lut", "h618")
    mats += load_matrix(ROOT / "results/t527_v4_formal_baked_expw/matrix_summary.csv", "lut", "t527")

    n_board = 0
    n_relu = 0
    for r in mats:
        new_us = relu_lat.get((r["device"], r["model"], r["dtype"]))
        r["lat_source"] = "original"
        if new_us is not None:
            r["latency_ms_p50"] = new_us / 1000.0
            r["lat_source"] = "b12_relu_rebuild"
            n_relu += 1
        acc_map = acc_f if r["arm"] == "flops" else acc_l
        r["test_acc_gpu"] = acc_map.get(r["key"])
        ba = board_acc.get((r["device"], r["model"])) or board_acc_any.get(r["model"])
        if ba:
            r.update(ba)
            r["board_acc_dtype"] = ba[f"board_{r['dtype']}_acc"]
            n_board += 1

        # Frontier/iso Y: §8.38 GPU distill test_acc (task accuracy);
        # board full-test accuracy is the deploy fidelity column.
        if r["test_acc_gpu"] is not None:
            r["plot_acc"] = float(r["test_acc_gpu"])
            r["acc_source"] = "gpu_distill_test_acc_s838"
        else:
            r["plot_acc"] = None
            r["acc_source"] = "missing"

    fields = [
        "arm", "device", "dataset", "dtype", "short", "key", "model",
        "latency_ms_p50", "lat_source", "test_acc_gpu", "plot_acc", "acc_source",
        "board_fp32_acc", "board_int8_acc", "board_fp32_int8_agree", "board_acc_dtype",
        "width", "depths", "img", "gray",
    ]
    with (OUT / "points.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(mats, key=lambda x: (x["device"], x["dataset"], x["arm"], x["dtype"], x["latency_ms_p50"])):
            w.writerow(r)

    iso_acc_rows = []
    iso_lat_rows = []
    for device in ("h618", "t527"):
        for dataset in ("ferplus", "expw"):
            lut = [
                r
                for r in mats
                if r["device"] == device
                and r["dataset"] == dataset
                and r["arm"] == "lut"
                and r["dtype"] == "int8"
                and r["plot_acc"] is not None
            ]
            flp = [
                r
                for r in mats
                if r["device"] == device
                and r["dataset"] == dataset
                and r["arm"] == "flops"
                and r["dtype"] == "int8"
                and r["plot_acc"] is not None
            ]
            if not lut or not flp:
                continue
            flp_by_acc = [(r["plot_acc"], r) for r in flp]
            flp_by_lat = [(r["latency_ms_p50"], r) for r in flp]
            for L in lut:
                a_near = nearest(L["plot_acc"], flp_by_acc, 0)[1]
                iso_acc_rows.append(
                    {
                        "device": device,
                        "dataset": dataset,
                        "lut_short": L["short"],
                        "lut_acc": L["plot_acc"],
                        "lut_lat_ms": L["latency_ms_p50"],
                        "flops_short": a_near["short"],
                        "flops_acc": a_near["plot_acc"],
                        "flops_lat_ms": a_near["latency_ms_p50"],
                        "d_acc": a_near["plot_acc"] - L["plot_acc"],
                        "lat_ratio_flops_over_lut": a_near["latency_ms_p50"] / L["latency_ms_p50"],
                        "acc_col": L["acc_source"],
                    }
                )
                l_near = nearest(L["latency_ms_p50"], flp_by_lat, 0)[1]
                iso_lat_rows.append(
                    {
                        "device": device,
                        "dataset": dataset,
                        "lut_short": L["short"],
                        "lut_lat_ms": L["latency_ms_p50"],
                        "lut_acc": L["plot_acc"],
                        "flops_short": l_near["short"],
                        "flops_lat_ms": l_near["latency_ms_p50"],
                        "flops_acc": l_near["plot_acc"],
                        "d_lat_ms": l_near["latency_ms_p50"] - L["latency_ms_p50"],
                        "d_acc": l_near["plot_acc"] - L["plot_acc"],
                        "acc_col": L["acc_source"],
                    }
                )

    def dump(name: str, rows: list[dict]) -> None:
        if not rows:
            (OUT / name).write_text("", encoding="utf-8")
            return
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("iso_acc.csv", iso_acc_rows)
    dump("iso_lat.csv", iso_lat_rows)

    apply()
    panels = [
        ("h618", "ferplus", "(a) H618, FER+"),
        ("t527", "ferplus", "(b) T527, FER+"),
        ("h618", "expw", "(c) H618, ExpW"),
        ("t527", "expw", "(d) T527, ExpW"),
    ]
    dev_color = {"h618": H618, "t527": T527}
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_IN, 4.15), sharey=True)
    ymin, ymax = 1.0, 0.0
    for ax, (device, dataset, label) in zip(axes.ravel(), panels):
        color = dev_color[device]
        for row in iso_acc_rows:
            if row["device"] == device and row["dataset"] == dataset:
                ax.plot(
                    [row["lut_lat_ms"], row["flops_lat_ms"]],
                    [row["lut_acc"], row["flops_acc"]],
                    color="0.75", lw=0.6, zorder=1,
                )
        for arm, marker, face in (("lut", "o", color), ("flops", "^", "none")):
            pts = [
                r for r in mats
                if r["device"] == device and r["dataset"] == dataset
                and r["arm"] == arm and r["dtype"] == "int8" and r["plot_acc"] is not None
            ]
            if not pts:
                continue
            xs = [r["latency_ms_p50"] for r in pts]
            ys = [r["plot_acc"] for r in pts]
            ymin, ymax = min(ymin, min(ys)), max(ymax, max(ys))
            ax.scatter(
                xs, ys, s=26, marker=marker, facecolors=face, edgecolors=color,
                linewidths=0.8, zorder=3,
            )
        ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=7)
    for ax in axes.ravel():
        ax.set_ylim(ymin - 0.015, ymax + 0.02)
    axes[1, 0].set_xlabel("P50 latency (ms)")
    axes[1, 1].set_xlabel("P50 latency (ms)")
    axes[0, 0].set_ylabel("Test accuracy")
    axes[1, 0].set_ylabel("Test accuracy")
    axes[0, 1].legend(
        handles=[
            Line2D([0], [0], marker="o", color=H618, ls="none", label="LUT"),
            Line2D([0], [0], marker="^", markerfacecolor="none", markeredgecolor=H618, color=H618, ls="none", label="FLOPs"),
        ],
        frameon=False, loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(OUT / "frontier_all_legacy.pdf")
    plt.close(fig)

    n_f = sum(1 for r in mats if r["arm"] == "flops")
    n_l = sum(1 for r in mats if r["arm"] == "lut")
    n_gpu = sum(1 for r in mats if str(r.get("acc_source", "")).startswith("gpu_distill"))
    note = [
        "# C1 FLOPs vs LUT — B-1/G-1 artifacts",
        "",
        f"- points: {len(mats)} (flops={n_f}, lut={n_l})",
        f"- iso_acc rows: {len(iso_acc_rows)}",
        f"- iso_lat rows: {len(iso_lat_rows)}",
        f"- frontier Y / 精度列: **§8.38 GPU distill test_acc** (gpu_src={n_gpu})",
        f"- board full-test accuracy filled: {n_board}; ReLU latency overrides (B12): {n_relu}",
        "- dtype plotted: NCNN INT8",
        "",
        "## Files",
        "- `points.csv`",
        "- `iso_acc.csv` (iso-accuracy nearest neighbor on task acc)",
        "- `iso_lat.csv` (iso-latency nearest neighbor)",
        "- `frontier_{h618,t527}_{ferplus,expw}_int8.png`",
        "",
    ]
    if iso_acc_rows:
        ratios = [r["lat_ratio_flops_over_lut"] for r in iso_acc_rows]
        note.append(
            f"## Iso-acc latency ratio (FLOPs/LUT): mean={np.mean(ratios):.3f} "
            f"median={np.median(ratios):.3f} min={np.min(ratios):.3f} max={np.max(ratios):.3f}"
        )
    (OUT / "README.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print(f"DONE -> {OUT} board_acc_filled={n_board} relu_lat_overrides={n_relu}")
    try:
        print((OUT / "README.md").read_text(encoding="utf-8"))
    except UnicodeEncodeError:
        print((OUT / "README.md").read_text(encoding="utf-8").encode("utf-8", "replace").decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
