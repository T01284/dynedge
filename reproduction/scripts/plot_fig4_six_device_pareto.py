#!/usr/bin/env python3
"""Fig.4: six-device accuracy--latency Pareto (FERPlus INT8 / static MCU).

Sources:
  - results/{h618,t527}_v4_formal_baked/matrix_summary.csv  (NCNN INT8)
  - results/orin_trt_int8_matrix.csv
  - lut/e2e_mcu_truth.csv (ESP32-S3 / P4)
  - GPU: accuracy-only formal mid (no wall-clock) — omitted from latency axis
  - Acc: formal_v4 / predistill / relu_mcu JSON test_acc

Outputs:
  results/paper_figures/fig4_six_device_pareto.{png,csv}
  paper/Neurocomputing/figures/fig4_six_device_pareto.png
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import H618, ORIN, P4, S3, T527, TEXT_IN, apply, save

OUT = ROOT / "results/paper_figures"
FIG = ROOT / "paper/Neurocomputing/figures"

DEVICE_STYLE = {
    "orin": {"color": ORIN, "marker": "o", "label": "Orin"},
    "t527": {"color": T527, "marker": "s", "label": "T527"},
    "h618": {"color": H618, "marker": "D", "label": "H618"},
    "esp32p4": {"color": P4, "marker": "^", "label": "ESP32-P4"},
    "esp32s3": {"color": S3, "marker": "v", "label": "ESP32-S3"},
}


def short_key(name: str) -> str | None:
    m = re.search(r"w([0-9.]+)_d(\d{4})_s(\d+)([gc])", name)
    if not m:
        return None
    return f"w{m.group(1)}_d{m.group(2)}_s{m.group(3)}{m.group(4)}"


def load_acc() -> dict[str, float]:
    """Prefer formal_lut seed42, then predistill, then relu_mcu."""
    buckets: dict[str, dict[str, float]] = {}
    roots = [
        ("formal", ROOT / "results/from_autodl/formal_v4/subnets_v4_formal"),
        ("predistill", ROOT / "results/from_autodl/formal_v4/subnets_v4_predistill"),
        ("predistill2", ROOT / "results/from_autodl/subnets_v4_predistill"),
        ("relu_mcu", ROOT / "results/from_autodl/formal_v4/subnets_v4_relu_mcu"),
    ]
    for tag, d in roots:
        if not d.exists():
            continue
        for p in d.glob("subnet_ferplus*.json"):
            if any(x in p.name for x in ("_s1234_", "_s2026_", "flops_ctrl")):
                continue
            key = short_key(p.name)
            if not key:
                continue
            try:
                acc = json.loads(p.read_text(encoding="utf-8")).get("test_acc")
            except Exception:
                continue
            if acc is None:
                continue
            buckets.setdefault(key, {})[tag] = float(acc)
    out: dict[str, float] = {}
    for key, m in buckets.items():
        for pref in ("formal", "predistill", "predistill2", "relu_mcu"):
            if pref in m:
                out[key] = m[pref]
                break
    return out


def lat_ms(row: dict) -> float | None:
    for k in ("latency_ms_p50", "p50_ms", "mean_latency_ms"):
        if row.get(k):
            return float(row[k])
    for k in ("latency_us_p50", "p50_us", "mean_latency_us"):
        if row.get(k):
            return float(row[k]) / 1000.0
    return None


def load_matrix_csv(path: Path, device: str, dtype_filter: str = "int8") -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            model = r.get("model") or r.get("stem") or ""
            dtype = (r.get("dtype") or r.get("backend") or "").lower()
            if dtype_filter and dtype_filter not in dtype and "trt_int8" not in dtype:
                if device != "orin":
                    continue
            key = short_key(model)
            ms = lat_ms(r)
            if not key or ms is None:
                continue
            rows.append({"device": device, "key": key, "model": model, "lat_ms": ms})
    return rows


def pareto_front(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(pts, key=lambda x: (x[0], -x[1]))
    front: list[tuple[float, float]] = []
    best = -1.0
    for lat, acc in pts:
        if acc > best + 1e-12:
            front.append((lat, acc))
            best = acc
    return front


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    acc = load_acc()

    points: list[dict] = []
    points += load_matrix_csv(ROOT / "results/h618_v4_formal_baked/matrix_summary.csv", "h618")
    points += load_matrix_csv(ROOT / "results/t527_v4_formal_baked/matrix_summary.csv", "t527")
    # Orin INT8
    for r in load_matrix_csv(ROOT / "results/orin_trt_int8_matrix.csv", "orin", dtype_filter=""):
        points.append(r)
    # MCU e2e
    mcu = ROOT / "lut/e2e_mcu_truth.csv"
    if mcu.exists():
        with mcu.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                dev = (r.get("device") or "").lower()
                if dev not in {"esp32s3", "esp32p4"}:
                    continue
                model = r.get("model") or ""
                key = short_key(model)
                ms = lat_ms(r)
                if not key or ms is None:
                    continue
                points.append({"device": dev, "key": key, "model": model, "lat_ms": ms})

    # attach acc
    kept = []
    for p in points:
        a = acc.get(p["key"])
        if a is None:
            continue
        p["acc"] = a
        kept.append(p)

    # write csv
    fields = ["device", "key", "model", "lat_ms", "acc"]
    with (OUT / "fig4_six_device_pareto.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(kept, key=lambda x: (x["device"], x["lat_ms"])))

    apply()
    fig, ax = plt.subplots(figsize=(TEXT_IN, 2.85), layout="constrained")
    for device, style in DEVICE_STYLE.items():
        pts = [p for p in kept if p["device"] == device]
        if not pts:
            continue
        xy = [(p["lat_ms"], p["acc"]) for p in pts]
        xs = [x[0] for x in xy]
        ys = [x[1] for x in xy]
        ax.scatter(
            xs, ys, c=style["color"], marker=style["marker"], s=28,
            label=f"{style['label']} (n={len(pts)})", zorder=3,
        )
        front = pareto_front(xy)
        if len(front) >= 2:
            fx, fy = zip(*front)
            ax.plot(fx, fy, "-", color=style["color"], lw=1.0, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("P50 latency (ms)")
    ax.set_ylabel("Test accuracy")
    fig.legend(frameon=False, ncol=5, loc="outside lower center")
    save(fig, "fig4_six_device_pareto")
    plt.close(fig)
    print(f"DONE points={len(kept)} devices={sorted({p['device'] for p in kept})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
