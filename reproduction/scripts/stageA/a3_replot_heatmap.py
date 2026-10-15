#!/usr/bin/env python3
"""Redraw the proxy heatmap from the saved rank table. Does not refit statistics."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import TEXT_IN, apply, save

COLS = [
    ("tau_flops_search", "FLOPs"),
    ("tau_macs_full", "MACs"),
    ("tau_params", "Params"),
    ("tau_n_layers", "Layers"),
    ("tau_act_elems", "Activations"),
    ("tau_lut_pred", "Op-LUT"),
]
SRC = Path(__file__).resolve().parents[2] / "results" / "stageA" / "a3_rank_fidelity.csv"


def main() -> int:
    apply()
    tab = pd.read_csv(SRC)
    mat = tab[[c for c, _ in COLS]].to_numpy(float)
    fig_h = max(3.2, 0.28 * len(tab) + 0.8)
    fig, ax = plt.subplots(figsize=(TEXT_IN, fig_h))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] < 0.55 else "black",
                )
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([lab for _, lab in COLS])
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels([f"{p} (n={int(n)})" for p, n in zip(tab.platform, tab.n)])
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Kendall $\\tau$")
    fig.tight_layout()
    save(fig, "fig_a3_heatmap")
    plt.close(fig)
    print(f"DONE rows={len(tab)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
