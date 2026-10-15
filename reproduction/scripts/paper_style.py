#!/usr/bin/env python3
"""Shared style for paper figures.

The submitted manuscript is elsarticle preprint, 12 pt. Its text width is 390 pt
(137 mm), which is the width a ``width=\\linewidth`` figure is printed at.
Figures are drawn at that size so 8 pt type stays 8 pt in the review PDF.
A published two-column page would be 90 mm per column and 190 mm full width;
switch full-width figures to ``figure*`` if the class option changes to 5p.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "Neurocomputing" / "figures"
# 390 pt / 72.27 pt per inch.
TEXT_IN = 390.0 / 72.27

# Okabe–Ito. One color per device, reused in every figure.
ORIN = "#E69F00"
T527 = "#0072B2"
H618 = "#009E73"
S3 = "#D55E00"
P4 = "#CC79A7"
INK = "#222222"
FLOPS = "#555555"

# Marker encodes the series when color is shared (device vs arm).
DEVICE_MARKER = {
    "orin": "o",
    "t527": "s",
    "h618": "D",
    "esp32s3": "v",
    "esp32p4": "^",
}


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    """Write PDF (the manuscript file) and a PNG preview."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
