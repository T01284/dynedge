#!/usr/bin/env python3
"""A-3: Intro motivation figure v1 — iso-FLOPs latency across ISA/devices.

Selects subnets whose FLOPs lie in dense ±5% clusters, then plots measured
wall-clock latency on H618 / T527 / Orin / ESP32-S3 / ESP32-P4 (MCU markers differ).

Data sources (Phase-1 measured):
  - results/{h618,t527}_v4_{formal,flops_ctrl}_baked/matrix_summary.csv
  - results/orin_trt{,_int8}_matrix.csv
  - lut/e2e_mcu_truth.csv

Outputs: results/motivation_flops_isa/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402

OUT = ROOT / "results/motivation_flops_isa"

DEVICE_ORDER = ["orin", "t527", "h618", "esp32p4", "esp32s3"]
DEVICE_LABEL = {
    "orin": "Orin (TRT)",
    "t527": "T527 (NCNN)",
    "h618": "H618 (NCNN)",
    "esp32p4": "ESP32-P4",
    "esp32s3": "ESP32-S3",
}
MARKERS = {
    "orin": "o",
    "t527": "s",
    "h618": "^",
    "esp32p4": "D",
    "esp32s3": "v",
}
COLORS = {
    "orin": "#1f77b4",
    "t527": "#2ca02c",
    "h618": "#ff7f0e",
    "esp32p4": "#9467bd",
    "esp32s3": "#d62728",
}


def parse_stem(s: str) -> SubnetConfig | None:
    m = re.search(r"w([0-9.]+)_d(\d{4})_s(\d+)([gc])", s)
    if not m:
        return None
    w_raw = m.group(1)
    if "." in w_raw:
        w = float(w_raw)
    elif w_raw.startswith("0") and len(w_raw) > 1:
        # w035 -> 0.35, w05 -> 0.5, w075 -> 0.75
        w = float("0." + w_raw[1:])
    else:
        w = float(w_raw)
    d = tuple(int(x) for x in m.group(2))
    img = int(m.group(3))
    gray = m.group(4) == "g"
    return SubnetConfig(width=w, depths=d, img_size=img, gray=gray)  # type: ignore[arg-type]


def cfg_key(cfg: SubnetConfig) -> str:
    g = "g" if cfg.gray else "c"
    d = "".join(str(x) for x in cfg.depths)
    return f"w{cfg.width:g}_d{d}_s{cfg.img_size}{g}"


def load_points(net: ElasticFERSuperNet) -> list[dict]:
    rows: list[dict] = []

    def add(device: str, backend: str, dtype: str, model: str, lat_ms: float, src: str) -> None:
        cfg = parse_stem(model)
        if cfg is None:
            return
        flops = float(net.estimate_flops(cfg))
        rows.append(
            {
                "device": device,
                "isa": {
                    "orin": "aarch64+GPU",
                    "t527": "A55",
                    "h618": "A53",
                    "esp32p4": "RISC-V",
                    "esp32s3": "Xtensa",
                }.get(device, "?"),
                "backend": backend,
                "dtype": dtype,
                "model": model,
                "cfg": cfg_key(cfg),
                "width": cfg.width,
                "depths": "".join(str(x) for x in cfg.depths),
                "img_size": cfg.img_size,
                "gray": int(cfg.gray),
                "flops": flops,
                "mflops": flops / 1e6,
                "lat_ms": lat_ms,
                "src": src,
            }
        )

    for device, path in (
        ("h618", ROOT / "results/h618_v4_formal_baked/matrix_summary.csv"),
        ("t527", ROOT / "results/t527_v4_formal_baked/matrix_summary.csv"),
        ("h618", ROOT / "results/h618_v4_flops_ctrl_baked/matrix_summary.csv"),
        ("t527", ROOT / "results/t527_v4_flops_ctrl_baked/matrix_summary.csv"),
    ):
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if "latency_ms_p50" in r and r["latency_ms_p50"]:
                lat = float(r["latency_ms_p50"])
            elif "latency_us_p50" in r and r["latency_us_p50"]:
                lat = float(r["latency_us_p50"]) / 1000.0
            else:
                continue
            add(device, "ncnn", r["dtype"], r["model"], lat, path.name)

    for path, dtype in (
        (ROOT / "results/orin_trt_matrix.csv", "fp16"),
        (ROOT / "results/orin_trt_int8_matrix.csv", "int8"),
    ):
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            add(
                "orin",
                "trt",
                dtype,
                r["model"],
                float(r["latency_us_p50"]) / 1000.0,
                path.name,
            )

    mcu = ROOT / "lut/e2e_mcu_truth.csv"
    if mcu.exists():
        for r in csv.DictReader(mcu.open(encoding="utf-8")):
            add(
                r["device"],
                "esp-dl",
                r.get("dtype") or "int8",
                r["model"],
                float(r["mean_latency_ms"]),
                "e2e_mcu_truth",
            )
    return rows


def prefer_dtype(rows: list[dict]) -> list[dict]:
    """One point per (device, cfg): prefer int8/esp-dl, else fp16/fp32."""
    best: dict[tuple[str, str], dict] = {}
    rank = {"int8": 0, "fp16": 1, "fp32": 2}

    def score(r: dict) -> tuple:
        return (rank.get(r["dtype"], 9), r["lat_ms"])

    for r in rows:
        k = (r["device"], r["cfg"])
        if k not in best or score(r) < score(best[k]):
            best[k] = r
    return list(best.values())


def pick_iso_cluster(
    rows: list[dict], tol: float = 0.05, min_cfgs: int = 8
) -> tuple[float, list[dict], float]:
    """Pick FLOPs center maximizing (#devices, #cfgs). Widen tol if needed."""
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["cfg"]].append(r)
    cfg_flops = {c: float(np.median([x["flops"] for x in xs])) for c, xs in by_cfg.items()}
    centers = sorted(set(cfg_flops.values()))

    def evaluate(tol_i: float):
        best = None
        for c in centers:
            lo, hi = c * (1 - tol_i), c * (1 + tol_i)
            members = [cfg for cfg, f in cfg_flops.items() if lo <= f <= hi]
            pts = [r for r in rows if r["cfg"] in members]
            devices = {r["device"] for r in pts}
            has_orin = 1 if "orin" in devices else 0
            has_mcu = 1 if (("esp32s3" in devices) or ("esp32p4" in devices)) else 0
            # Prefer device coverage near MCU-anchor FLOPs (~1–3 MFLOPs)
            score = (
                len(devices),
                has_orin,
                has_mcu,
                len(members),
                -abs(np.log(c + 1.0) - np.log(2e6)),
            )
            if best is None or score > best[0]:
                best = (score, c, members, pts, tol_i)
        return best

    best = evaluate(tol)
    assert best is not None
    for tol_i in (0.08, 0.12, 0.15, 0.20, 0.30, 0.50):
        if len(best[2]) >= min_cfgs and best[0][0] >= 4:
            break
        cand = evaluate(tol_i)
        if cand is None:
            continue
        # Accept if more cfgs while keeping ≥4 devices, or strictly better device coverage
        if cand[0][0] > best[0][0] or (
            cand[0][0] >= best[0][0] and len(cand[2]) > len(best[2])
        ):
            best = cand
    _, center, _members, pts, used_tol = best
    return center, pts, used_tol


def matched_multidevice(rows: list[dict], min_devices: int = 3) -> list[dict]:
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["cfg"]].append(r)
    out: list[dict] = []
    for _cfg, xs in by_cfg.items():
        if len({x["device"] for x in xs}) >= min_devices:
            out.extend(xs)
    return out


def plot_matched(pts: list[dict], out_png: Path, out_pdf: Path) -> None:
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in pts:
        by_cfg[r["cfg"]].append(r)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    devices = [d for d in DEVICE_ORDER if any(r["device"] == d for r in pts)]
    xpos = {d: i for i, d in enumerate(devices)}
    cmap = plt.get_cmap("tab20")
    for i, (_cfg, xs) in enumerate(sorted(by_cfg.items(), key=lambda kv: kv[1][0]["flops"])):
        xs_sorted = sorted(xs, key=lambda r: xpos[r["device"]])
        ax.plot(
            [xpos[r["device"]] for r in xs_sorted],
            [r["lat_ms"] for r in xs_sorted],
            "-o",
            color=cmap(i % 20),
            lw=1.2,
            ms=5,
            alpha=0.85,
            label=_cfg,
        )
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([DEVICE_LABEL[d] for d in devices], rotation=15, ha="right")
    ax.set_ylabel("Measured latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Identical subnet configs: latency diverges across ISA")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_full_scatter(pts: list[dict], out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for device in DEVICE_ORDER:
        xs = [r["mflops"] for r in pts if r["device"] == device]
        ys = [r["lat_ms"] for r in pts if r["device"] == device]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=48 if device.startswith("esp") else 36,
            marker=MARKERS[device],
            c=COLORS[device],
            label=DEVICE_LABEL[device],
            alpha=0.85,
            edgecolors="k",
            linewidths=0.35,
            zorder=3 if device.startswith("esp") else 2,
        )
    ax.set_xlabel("FLOPs (MFLOPs)")
    ax.set_ylabel("Measured latency (ms)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("FLOPs is a poor latency proxy across ISA / backends")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_scatter(pts: list[dict], center: float, out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    # x = FLOPs (MFLOPs), y = latency (ms), color/marker by device
    for device in DEVICE_ORDER:
        xs = [r["mflops"] for r in pts if r["device"] == device]
        ys = [r["lat_ms"] for r in pts if r["device"] == device]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=48 if device.startswith("esp") else 36,
            marker=MARKERS[device],
            c=COLORS[device],
            label=DEVICE_LABEL[device],
            alpha=0.9,
            edgecolors="k",
            linewidths=0.4,
            zorder=3 if device.startswith("esp") else 2,
        )
    lo, hi = center * 0.95 / 1e6, center * 1.05 / 1e6
    ax.axvspan(lo, hi, color="#bbbbbb", alpha=0.25, label="±5% FLOPs band")
    ax.set_xlabel("FLOPs (MFLOPs)")
    ax.set_ylabel("Measured latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Same FLOPs ≠ same latency across ISA / backends")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_device_strip(pts: list[dict], center: float, out_png: Path, out_pdf: Path) -> None:
    """Strip / swarm: latency distribution per device within iso-FLOPs cluster."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    devices = [d for d in DEVICE_ORDER if any(r["device"] == d for r in pts)]
    rng = np.random.default_rng(42)
    for i, device in enumerate(devices):
        ys = np.array([r["lat_ms"] for r in pts if r["device"] == device], dtype=float)
        xs = i + rng.uniform(-0.18, 0.18, size=len(ys))
        ax.scatter(
            xs,
            ys,
            s=50 if device.startswith("esp") else 36,
            marker=MARKERS[device],
            c=COLORS[device],
            edgecolors="k",
            linewidths=0.4,
            label=DEVICE_LABEL[device],
            zorder=3,
        )
        if len(ys):
            ax.hlines(float(np.median(ys)), i - 0.28, i + 0.28, colors="k", lw=1.6, zorder=4)
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([DEVICE_LABEL[d] for d in devices], rotation=15, ha="right")
    ax.set_ylabel("Measured latency (ms)")
    ax.set_yscale("log")
    ax.set_title(
        f"Iso-FLOPs cluster (around {center/1e6:.1f} MFLOPs): latency by device"
    )
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--min-cfgs", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    net = ElasticFERSuperNet(num_classes=8, slimmable=True)
    raw = load_points(net)
    # Drop pathological parses / outliers (e.g. width misread as 35)
    raw = [r for r in raw if 0.1 <= r["mflops"] <= 200.0]
    pts = prefer_dtype(raw)
    center, cluster, used_tol = pick_iso_cluster(pts, tol=args.tol, min_cfgs=args.min_cfgs)
    matched = matched_multidevice(pts, min_devices=3)

    # write CSVs
    keys = list(pts[0].keys()) if pts else []
    with (OUT / "all_points.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(pts)
    with (OUT / "iso_flops_cluster.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(cluster)
    with (OUT / "matched_multidevice.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(matched)

    plot_full_scatter(
        pts,
        OUT / "motivation_flops_vs_latency_all.png",
        OUT / "motivation_flops_vs_latency_all.pdf",
    )
    plot_scatter(
        cluster,
        center,
        OUT / "motivation_flops_vs_latency.png",
        OUT / "motivation_flops_vs_latency.pdf",
    )
    plot_device_strip(
        cluster,
        center,
        OUT / "motivation_iso_flops_by_device.png",
        OUT / "motivation_iso_flops_by_device.pdf",
    )
    if matched:
        plot_matched(
            matched,
            OUT / "motivation_matched_cfgs.png",
            OUT / "motivation_matched_cfgs.pdf",
        )

    # stats
    by_dev = {
        d: [r["lat_ms"] for r in cluster if r["device"] == d]
        for d in DEVICE_ORDER
        if any(r["device"] == d for r in cluster)
    }
    stats = {
        "center_flops": center,
        "center_mflops": center / 1e6,
        "tol_requested": args.tol,
        "tol_used": used_tol,
        "n_points": len(cluster),
        "n_cfgs": len({r["cfg"] for r in cluster}),
        "matched_n_cfgs": len({r["cfg"] for r in matched}),
        "matched_n_points": len(matched),
        "devices": {
            d: {
                "n": len(v),
                "lat_min_ms": min(v),
                "lat_max_ms": max(v),
                "lat_median_ms": float(np.median(v)),
                "lat_span_x": (max(v) / min(v)) if min(v) > 0 else None,
            }
            for d, v in by_dev.items()
            if v
        },
        "cross_device_span_x": (
            max(max(v) for v in by_dev.values()) / min(min(v) for v in by_dev.values())
            if by_dev and all(v for v in by_dev.values())
            else None
        ),
        "cfgs": sorted({r["cfg"] for r in cluster}),
        "matched_cfgs": sorted({r["cfg"] for r in matched}),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
