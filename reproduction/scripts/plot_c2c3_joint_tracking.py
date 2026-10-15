#!/usr/bin/env python3
"""C2+C3 joint budget-tracking money figure (host synthesis).

X: target budget B ∈ [3, 30] ms
Y: achieved E[lat]

C2 (coarse): among catalogued subnets with Lat_lo ≤ B, pick highest test_acc
            (static formal = single-exit board FP32; elastic C3 subnet = exit2..full).
C3 (fine):  when the elastic subnet is selected, interpolate §8.36 randB λ=0.05
            E[lat](B) within its band; static subnets stay at full latency.

Also emits C2-only staircase (E[lat]=Lat_hi) for contrast.

Demotes the old single-subnet three-line plot to a 4.4 mechanism figure.

Examples:
  py -3 scripts/plot_c2c3_joint_tracking.py
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c2c3_joint_tracking"
MECH = ROOT / "results/c3_budget_tracking/mechanism_4_4"
TRACK_CSV = ROOT / "results/c3_budget_tracking/budget_tracking.csv"
FORMAL_JSON = ROOT / "results/from_autodl/formal_v4/subnets_v4_formal"
MAT = {
    "h618": ROOT / "results/h618_v4_formal_baked/matrix_summary.csv",
    "t527": ROOT / "results/t527_v4_formal_baked/matrix_summary.csv",
}
FEED = {
    "h618": ROOT / "results/h618_v4/exit_segment_latency/summary.json",
    "t527": ROOT / "results/t527_v4/exit_segment_latency/summary.json",
}
DEVICES = ("h618", "t527")
B_LO, B_HI = 3.0, 30.0
N_SWEEP = 400


@dataclass
class Subnet:
    name: str
    test_acc: float
    lo_ms: float  # earliest usable (exit2 or full for static)
    hi_ms: float  # full-net
    elastic: bool = False
    # sorted (B, E[lat]) for elastic C3 response; empty → E[lat]=hi
    c3_curve: list[tuple[float, float]] = field(default_factory=list)

    def e_lat(self, B: float) -> float:
        if not self.elastic or not self.c3_curve:
            return float(self.hi_ms)
        if B >= self.hi_ms - 1e-9:
            return float(self.hi_ms)
        bs = np.array([p[0] for p in self.c3_curve], dtype=np.float64)
        ys = np.array([p[1] for p in self.c3_curve], dtype=np.float64)
        if B <= bs[0]:
            return float(ys[0])
        if B >= bs[-1]:
            return float(ys[-1])
        return float(np.interp(B, bs, ys))


def _stem_key(model: str) -> str:
    m = re.search(r"(subnet_ferplus_[\w.]+_v4_formal_lut)", model)
    return m.group(1) if m else model


def load_formal_acc() -> dict[str, float]:
    out: dict[str, float] = {}
    for p in FORMAL_JSON.glob("subnet_ferplus_*_v4_formal_lut.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        key = p.stem  # ..._v4_formal_lut
        out[key] = float(d["test_acc"])
    return out


def load_matrix_fp32(path: Path) -> dict[str, float]:
    """stem -> p50 ms."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: dict[str, float] = {}
    for r in rows:
        if r.get("dtype", "fp32") != "fp32":
            continue
        model = r["model"]
        if "latency_ms_p50" in r and r["latency_ms_p50"]:
            ms = float(r["latency_ms_p50"])
        else:
            ms = float(r["latency_us_p50"]) / 1000.0
        # strip img suffix variants for key match
        stem = model
        for suf in ("_img64", "_img96", "_img112"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        out[stem] = ms
    return out


def load_c3_randb(device: str) -> tuple[float, float, list[tuple[float, float]], float]:
    """Return (lo, hi, curve[(B,lat)], mid_acc) from §8.36 λ=0.05 randB."""
    rows = list(csv.DictReader(TRACK_CSV.open(encoding="utf-8")))
    sub = [
        r
        for r in rows
        if r["device"] == device and r["kind"] == "c3_randB_l0p05"
    ]
    if not sub:
        raise SystemExit(f"no c3_randB_l0p05 rows for {device}")
    sub = sorted(sub, key=lambda r: float(r["budget_ms"]))
    lo = float(sub[0]["band_lo"])
    hi = float(sub[0]["full_ms"])
    curve = [(float(r["budget_ms"]), float(r["lat_ms"])) for r in sub]
    acc = float(np.mean([float(r["acc"]) for r in sub]))
    return lo, hi, curve, acc


def build_catalog(device: str) -> list[Subnet]:
    accs = load_formal_acc()
    mats = load_matrix_fp32(MAT[device])
    cats: list[Subnet] = []
    for stem, ms in sorted(mats.items(), key=lambda kv: kv[1]):
        # match json stem
        acc = None
        for k, a in accs.items():
            if k in stem or stem.startswith(k):
                acc = a
                break
        if acc is None:
            # fuzzy: width/depth/img tokens
            continue
        short = stem.replace("subnet_ferplus_", "").replace("_seed42_v4_formal_lut", "")
        cats.append(
            Subnet(
                name=f"static:{short}",
                test_acc=acc,
                lo_ms=ms,
                hi_ms=ms,
                elastic=False,
            )
        )
    lo, hi, curve, c3_acc = load_c3_randb(device)
    cats.append(
        Subnet(
            name="elastic:w0.5_d3333_s112+C3",
            test_acc=c3_acc,
            lo_ms=lo,
            hi_ms=hi,
            elastic=True,
            c3_curve=curve,
        )
    )
    # also force_final point of elastic as upper static-like (already hi)
    return cats


# Acc slack for budget-filling among near-optimal subnets (percentage points).
# Pure max-Acc would stick on the ~9.5ms formal mid and never enter the C3 band.
ACC_SLACK = 0.015


def select_c2(catalog: list[Subnet], B: float) -> Subnet | None:
    """C2: Lat_lo ≤ B; max Acc, then fill budget among Acc within ACC_SLACK."""
    feas = [s for s in catalog if s.lo_ms <= B + 1e-9]
    if not feas:
        return None
    best_acc = max(s.test_acc for s in feas)
    near = [s for s in feas if s.test_acc >= best_acc - ACC_SLACK]
    # Prefer using the budget: maximize min(hi, B), then Acc, then elastic
    near.sort(
        key=lambda s: (min(s.hi_ms, B), s.test_acc, 1 if s.elastic else 0),
        reverse=True,
    )
    return near[0]


def sweep(device: str) -> list[dict]:
    catalog = build_catalog(device)
    Bs = np.linspace(B_LO, B_HI, N_SWEEP)
    rows: list[dict] = []
    for B in Bs:
        s = select_c2(catalog, float(B))
        if s is None:
            rows.append(
                {
                    "device": device,
                    "budget_ms": float(B),
                    "e_lat_c2c3": float("nan"),
                    "e_lat_c2_only": float("nan"),
                    "subnet": "",
                    "elastic": 0,
                    "test_acc": float("nan"),
                    "lo_ms": float("nan"),
                    "hi_ms": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "device": device,
                "budget_ms": float(B),
                "e_lat_c2c3": s.e_lat(float(B)),
                "e_lat_c2_only": float(s.hi_ms),
                "subnet": s.name,
                "elastic": int(s.elastic),
                "test_acc": s.test_acc,
                "lo_ms": s.lo_ms,
                "hi_ms": s.hi_ms,
            }
        )
    return rows


def plot_money(all_rows: list[dict], catalogs: dict[str, list[Subnet]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=False)
    titles = {"h618": "H618 (NCNN CPU)", "t527": "T527 (NCNN CPU)"}
    colors = {"h618": "#1f77b4", "t527": "#ff7f0e"}

    for ax, dev in zip(axes, DEVICES):
        rows = [r for r in all_rows if r["device"] == dev]
        c = colors[dev]
        B = np.array([r["budget_ms"] for r in rows])
        y_j = np.array([r["e_lat_c2c3"] for r in rows])
        y_c2 = np.array([r["e_lat_c2_only"] for r in rows])

        ax.plot([B_LO, B_HI], [B_LO, B_HI], "--", color="0.55", lw=1.4, label="ideal $y=x$", zorder=1)
        # C2-only stairs (slightly behind)
        ax.plot(B, y_c2, "-", color="0.65", lw=1.6, alpha=0.9, label="C2 only (static full)", zorder=2)
        ax.plot(B, y_j, "-", color=c, lw=2.2, label="C2 + C3 (joint)", zorder=3)

        # mark elastic band
        cat = catalogs[dev]
        el = next(s for s in cat if s.elastic)
        ax.axvspan(el.lo_ms, min(el.hi_ms, B_HI), color=c, alpha=0.08, zorder=0, label="C3 band (elastic)")

        # annotate static step levels (unique hi)
        seen = set()
        for s in sorted(cat, key=lambda x: x.hi_ms):
            if s.elastic:
                continue
            if s.hi_ms > B_HI + 0.5:
                continue
            key = round(s.hi_ms, 2)
            if key in seen:
                continue
            seen.add(key)
            ax.axhline(s.hi_ms, color="0.85", lw=0.7, zorder=0)

        ax.set_xlim(B_LO, B_HI)
        ax.set_ylim(0, B_HI + 1)
        ax.set_xlabel("target budget $B$ (ms)")
        ax.set_title(titles[dev])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    axes[0].set_ylabel(r"achieved $E[\mathrm{lat}]$ (ms)")
    fig.suptitle(
        r"C2+C3 joint latency tracking — coarse subnet select + in-band early-exit",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"c2c3_joint_tracking.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    # single overlay panel
    fig2, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot([B_LO, B_HI], [B_LO, B_HI], "--", color="0.5", lw=1.4, label="ideal $y=x$")
    for dev in DEVICES:
        rows = [r for r in all_rows if r["device"] == dev]
        B = [r["budget_ms"] for r in rows]
        y = [r["e_lat_c2c3"] for r in rows]
        ax.plot(B, y, "-", color=colors[dev], lw=2.0, label=titles[dev])
    ax.set_xlim(B_LO, B_HI)
    ax.set_ylim(0, B_HI + 1)
    ax.set_xlabel("target budget $B$ (ms)")
    ax.set_ylabel(r"achieved $E[\mathrm{lat}]$ (ms)")
    ax.set_title("C2+C3 joint tracking (H618 / T527)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(out / f"c2c3_joint_tracking_overlay.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig2)


def _plot_c3_mechanism(rows: list[dict], out_dir: Path) -> None:
    """§4.4 single-subnet ideal/band/randB (no torch)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = ("h618", "t527", "orin")
    colors = {"h618": "#1f77b4", "t527": "#ff7f0e", "orin": "#2ca02c"}
    titles = {"h618": "H618 (NCNN)", "t527": "T527 (NCNN)", "orin": "Orin (TRT seg)"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=False)
    for ax, dev in zip(axes, devices):
        any_row = [r for r in rows if r["device"] == dev]
        if not any_row:
            ax.set_visible(False)
            continue
        lo = float(any_row[0]["band_lo"])
        hi = float(any_row[0]["full_ms"])
        pad = 0.05 * (hi - lo + 1e-9)
        lim_lo, lim_hi = lo - pad, hi + pad
        c = colors[dev]
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="0.55", lw=1.3, label="ideal $B$", zorder=1)
        ax.axvspan(lo, hi, color=c, alpha=0.07, zorder=0)
        band = sorted(
            [r for r in rows if r["device"] == dev and r["kind"] == "c3_band"],
            key=lambda r: float(r["budget_ms"]),
        )
        if band:
            ax.plot(
                [float(r["budget_ms"]) for r in band],
                [float(r["lat_ms"]) for r in band],
                "-s",
                color=c,
                ms=6,
                lw=2.0,
                label=r"band specialist $\lambda$=0.2",
                zorder=3,
            )
        kinds = {r["kind"] for r in rows if r["device"] == dev and str(r["kind"]).startswith("c3_randB")}
        rk = next((k for k in ("c3_randB_l0p05", "c3_randB_l0p2") if k in kinds), None)
        if rk:
            sub = sorted(
                [r for r in rows if r["device"] == dev and r["kind"] == rk],
                key=lambda r: float(r["budget_ms"]),
            )
            ax.plot(
                [float(r["budget_ms"]) for r in sub],
                [float(r["lat_ms"]) for r in sub],
                ":o",
                color="0.2",
                ms=4,
                lw=1.5,
                label=rf"randB $\lambda$={sub[0].get('lambda_lat','')}",
                zorder=2,
            )
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("target budget $B$ (ms)")
        ax.set_title(titles[dev])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, framealpha=0.92)
    axes[0].set_ylabel(r"achieved $E[\mathrm{lat}]$ (ms)")
    fig.tight_layout()
    for name in ("c3_mechanism_band_vs_randB_3dev", "budget_tracking_3dev"):
        # write mechanism name into MECH; also keep legacy name in parent if out_dir is MECH parent
        pass
    fig.savefig(out_dir / "c3_mechanism_band_vs_randB_3dev.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "c3_mechanism_band_vs_randB_3dev.pdf", bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1.3, label="ideal")
    for dev in devices:
        any_row = [r for r in rows if r["device"] == dev]
        if not any_row:
            continue
        lo = float(any_row[0]["band_lo"])
        hi = float(any_row[0]["full_ms"])
        span = max(hi - lo, 1e-9)
        c = colors[dev]
        band = sorted(
            [r for r in rows if r["device"] == dev and r["kind"] == "c3_band"],
            key=lambda r: float(r["budget_ms"]),
        )
        if band:
            bn = (np.array([float(r["budget_ms"]) for r in band]) - lo) / span
            ln = (np.array([float(r["lat_ms"]) for r in band]) - lo) / span
            ax.plot(bn, ln, "-s", color=c, ms=5, lw=1.8, label=f"{titles[dev]} band")
        kinds = {r["kind"] for r in rows if r["device"] == dev and str(r["kind"]).startswith("c3_randB")}
        rk = next((k for k in ("c3_randB_l0p05", "c3_randB_l0p2") if k in kinds), None)
        if rk:
            sub = sorted(
                [r for r in rows if r["device"] == dev and r["kind"] == rk],
                key=lambda r: float(r["budget_ms"]),
            )
            bn = (np.array([float(r["budget_ms"]) for r in sub]) - lo) / span
            ln = (np.array([float(r["lat_ms"]) for r in sub]) - lo) / span
            ax.plot(bn, ln, ":o", color=c, ms=4, lw=1.2, alpha=0.85, label=f"{titles[dev]} randB")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"normalized budget $(B-B_{\min})/(B_{\max}-B_{\min})$")
    ax.set_ylabel(r"normalized $E[\mathrm{lat}]$")
    ax.set_title("C3 mechanism — normalized (paper §4.4)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig2.tight_layout()
    fig2.savefig(out_dir / "c3_mechanism_normalized.png", dpi=200, bbox_inches="tight")
    fig2.savefig(out_dir / "c3_mechanism_normalized.pdf", bbox_inches="tight")
    plt.close(fig2)


def demote_mechanism_figs() -> None:
    """Regenerate single-subnet three-line plots into 4.4 mechanism folder."""
    MECH.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(TRACK_CSV.open(encoding="utf-8")))
    _plot_c3_mechanism(rows, MECH)
    # Keep legacy filenames in c3_budget_tracking/ for old links, with mechanism caption elsewhere.
    src = ROOT / "results/c3_budget_tracking"
    for a, b in (
        ("c3_mechanism_band_vs_randB_3dev.png", "budget_tracking_3dev.png"),
        ("c3_mechanism_band_vs_randB_3dev.pdf", "budget_tracking_3dev.pdf"),
        ("c3_mechanism_normalized.png", "budget_tracking_normalized.png"),
        ("c3_mechanism_normalized.pdf", "budget_tracking_normalized.pdf"),
    ):
        p = MECH / a
        if p.is_file():
            shutil.copy2(p, src / b)
    (MECH / "README.md").write_text(
        "# C3 mechanism analysis (paper §4.4)\n\n"
        "Former A-1 three-line plot (ideal / band / randB on a **single** "
        "subnet). Slope ≪ 1 is structural: only 3 discrete exits; H618 "
        "E[lat] span ≈21.4→24.1 ms. Coarse adaptation is C2 — system money "
        "figure is `results/c2c3_joint_tracking/`.\n",
        encoding="utf-8",
    )


def tracking_stats(rows: list[dict]) -> dict:
    """Rough MAE vs ideal on finite points."""
    out = {}
    for dev in DEVICES:
        sub = [r for r in rows if r["device"] == dev and r["subnet"]]
        B = np.array([r["budget_ms"] for r in sub])
        y = np.array([r["e_lat_c2c3"] for r in sub])
        y2 = np.array([r["e_lat_c2_only"] for r in sub])
        mae = float(np.nanmean(np.abs(y - B)))
        mae2 = float(np.nanmean(np.abs(y2 - B)))
        # correlation / slope of joint vs B (piecewise; report overall OLS)
        if len(B) > 2:
            slope = float(np.polyfit(B, y, 1)[0])
            slope2 = float(np.polyfit(B, y2, 1)[0])
        else:
            slope = slope2 = float("nan")
        out[dev] = {
            "mae_joint_ms": round(mae, 3),
            "mae_c2_only_ms": round(mae2, 3),
            "ols_slope_joint": round(slope, 3),
            "ols_slope_c2_only": round(slope2, 3),
            "n": len(sub),
        }
    return out


def main() -> int:
    catalogs = {d: build_catalog(d) for d in DEVICES}
    all_rows: list[dict] = []
    for d in DEVICES:
        all_rows.extend(sweep(d))

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "joint_tracking_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    cat_dump = {
        d: [
            {
                "name": s.name,
                "test_acc": s.test_acc,
                "lo_ms": s.lo_ms,
                "hi_ms": s.hi_ms,
                "elastic": s.elastic,
                "n_c3_pts": len(s.c3_curve),
            }
            for s in catalogs[d]
        ]
        for d in DEVICES
    }
    stats = tracking_stats(all_rows)
    summary = {
        "status": "ok_c2c3_joint_money_figure",
        "note": (
            "Host synthesis: C2 = Lat_lo≤B, max Acc then budget-fill among "
            f"Acc within ±{ACC_SLACK:.0%} (formal FP32 + elastic C3 mid); "
            "C3 fine-tune = §8.36 randB λ=0.05. "
            "Single-subnet three-line plot demoted to mechanism_4_4/ (§4.4)."
        ),
        "c2_acc_slack": ACC_SLACK,
        "B_range_ms": [B_LO, B_HI],
        "catalog": cat_dump,
        "stats": stats,
        "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "figures": [
            "results/c2c3_joint_tracking/c2c3_joint_tracking.png",
            "results/c2c3_joint_tracking/c2c3_joint_tracking_overlay.png",
        ],
        "mechanism_4_4": "results/c3_budget_tracking/mechanism_4_4/",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_money(all_rows, catalogs, OUT)
    demote_mechanism_figs()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
