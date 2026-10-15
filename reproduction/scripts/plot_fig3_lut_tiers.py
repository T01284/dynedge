#!/usr/bin/env python3
"""Fig.3: latency modeling tiers + H618 operator-LUT fidelity scatter.

Uses lut/fidelity_h618_v4.json (r=0.996, MAPE=5.3%).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import H618, INK, ORIN, S3, TEXT_IN, apply, save

FID = ROOT / "lut/fidelity_h618_v4.json"


def main() -> int:
    apply()
    d = json.loads(FID.read_text(encoding="utf-8"))
    xs = np.array([r["measured_us"] / 1000.0 for r in d["rows"]], dtype=float)
    ys = np.array([r["pred_us"] / 1000.0 for r in d["rows"]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.45))

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    tiers = [
        (6.7, H618, "Operator LUT\nH618 and T527 CPU"),
        (3.5, ORIN, "End-to-end points\nOrin TensorRT, T527 NPU"),
        (0.3, S3, "Exhaustive table\nESP32-S3 and ESP32-P4"),
    ]
    for y, c, text in tiers:
        ax.add_patch(plt.Rectangle((0.4, y), 9.2, 2.8, fill=False, edgecolor=c, lw=1.0))
        ax.text(5.0, y + 1.4, text, ha="center", va="center", fontsize=8, color=INK)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")

    ax = axes[1]
    lo = min(xs.min(), ys.min()) * 0.95
    hi = max(xs.max(), ys.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="0.45", lw=0.8, zorder=1)
    ax.scatter(xs, ys, s=22, marker="D", facecolors="none", edgecolors=H618, linewidths=0.8, zorder=2)
    ax.set_xlabel("Measured P50 (ms)")
    ax.set_ylabel("LUT prediction (ms)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.04, 0.96,
        f"(b)\n$r$ = {d['pearson_r']:.3f}\nMAPE = {d['mape_pct']:.1f}%\nn = {len(xs)}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK,
    )

    fig.tight_layout()
    save(fig, "fig3_lut_tiers")
    plt.close(fig)
    print(f"DONE n={len(xs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
