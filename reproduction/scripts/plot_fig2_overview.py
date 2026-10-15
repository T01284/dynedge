#!/usr/bin/env python3
"""Fig. 2: study design (supernet, five devices, three questions).

Drawn at the preprint text width so labels stay at least 7 pt in the review PDF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import H618, INK, ORIN, P4, S3, T527, TEXT_IN, apply, save


def box(ax, x, y, w, h, edge, face="#ffffff"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        linewidth=0.8, edgecolor=edge, facecolor=face, mutation_aspect=0.4, zorder=2,
    ))


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
        linewidth=0.7, color=INK, shrinkA=0, shrinkB=0, zorder=1,
    ))


def main() -> int:
    apply()
    fig, ax = plt.subplots(figsize=(TEXT_IN, 3.55))
    ax.set_xlim(0, TEXT_IN)
    ax.set_ylim(0, 3.55)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Top: supernet -> sampled subnets -> what is measured
    top = [
        (0.08, "Supernet", "Elastic MobileNetV3\nwidth, depth, resolution\nFER+ and ExpW\nGPU server: train, accuracy"),
        (1.95, "Sampled subnets", "972-candidate space\ndistilled students\nper tier"),
        (3.82, "What is measured", "Latency, P50\nFull-test accuracy\nIdle-subtracted energy"),
    ]
    for x, title, body in top:
        box(ax, x, 2.45, 1.55, 0.95, INK)
        ax.text(x + 0.78, 3.22, title, ha="center", va="top", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + 0.78, 2.72, body, ha="center", va="center", fontsize=6.5, color=INK, linespacing=1.25)
    arrow(ax, 1.63, 2.92, 1.95, 2.92)
    arrow(ax, 3.50, 2.92, 3.82, 2.92)

    # Five devices
    devices = [
        ("Orin", "TensorRT", "FP16, INT8", ORIN),
        ("T527", "NCNN and NPU", "FP32, INT8, NBG", T527),
        ("H618", "NCNN", "FP32, INT8", H618),
        ("ESP32-P4", "ESP-DL", "INT8", P4),
        ("ESP32-S3", "ESP-DL", "INT8", S3),
    ]
    gap = 0.08
    left = 0.08
    w = (TEXT_IN - 2 * left - 4 * gap) / 5
    y, h = 1.28, 0.95
    for i, (name, backend, fmt, color) in enumerate(devices):
        x = left + i * (w + gap)
        box(ax, x, y, w, h, color)
        ax.text(x + w / 2, y + h - 0.12, name, ha="center", va="top", fontsize=7.5, fontweight="bold", color=color)
        ax.text(x + w / 2, y + 0.42, backend, ha="center", va="center", fontsize=6.5, color=INK)
        ax.text(x + w / 2, y + 0.16, fmt, ha="center", va="center", fontsize=6.5, color=INK)
    # stem from the measurement box down onto the device row
    ax.plot([4.60, 4.60], [2.45, 2.28], color=INK, lw=0.7, zorder=1)
    ax.plot([left + w / 2, left + 4 * (w + gap) + w / 2], [2.28, 2.28], color=INK, lw=0.7, zorder=1)
    for i in range(5):
        x = left + i * (w + gap) + w / 2
        arrow(ax, x, 2.28, x, y + h)

    # Three questions
    qs = [
        ("Q1", "Where does FLOPs\nrank latency?"),
        ("Q2", "Does measuring change\nthe searched front?"),
        ("Q3", "Does a learned controller\nbeat a threshold?"),
    ]
    qw = (TEXT_IN - 2 * left - 2 * gap) / 3
    for i, (q, body) in enumerate(qs):
        x = left + i * (qw + gap)
        box(ax, x, 0.10, qw, 0.95, INK, face="#f7f7f7")
        ax.text(x + 0.12, 0.86, q, ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + qw / 2, 0.42, body, ha="center", va="center", fontsize=7, color=INK, linespacing=1.25)
    xs = [left + i * (w + gap) + w / 2 for i in range(5)]
    bus = 1.18
    ax.plot([xs[0], xs[-1]], [bus, bus], color=INK, lw=0.7, zorder=1)
    for x in xs:
        ax.plot([x, x], [y, bus], color=INK, lw=0.7, zorder=1)
    arrow(ax, xs[2], bus, xs[2], 1.07)

    save(fig, "fig2_overview")
    plt.close(fig)
    print("DONE fig2_overview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
