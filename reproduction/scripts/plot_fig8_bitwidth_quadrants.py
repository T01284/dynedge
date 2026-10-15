#!/usr/bin/env python3
"""Fig.8: INT8 vs FP32 latency--energy quadrants (H618 / T527 / Orin).

For each matched (model, device):
  x = latency_int8 / latency_fp32   (Orin: INT8 / FP16)
  y = energy_int8_net / energy_fp32_net

Quadrants relative to (1,1):
  lower-left  = INT8 faster & thriftier
  upper-right = INT8 slower & hungrier
  etc.

Sources:
  results/power/{h618,t527,orin}_p5s1/energy_matrix.csv
  matching NCNN/TRT latency from bake / orin matrices

Outputs:
  results/paper_figures/fig8_bitwidth_quadrants.{png,csv}
  paper/Neurocomputing/figures/fig8_bitwidth_quadrants.png
"""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_figures"
FIG = ROOT / "paper/Neurocomputing/figures"


def short_key(name: str) -> str | None:
    m = re.search(r"w([0-9.]+)_d(\d{4})_s(\d+)([gc])|w0?(\d+)_d(\d{4})_s(\d+)([gc])", name)
    if not m:
        # energy tags like w05_d3333_s112c
        m = re.search(r"w0?(\d+)_d(\d{4})_s(\d+)([gc])", name)
        if not m:
            return None
        wraw, d, img, gc = m.groups()
        if wraw.startswith("0") and len(wraw) > 1:
            w = "0." + wraw[1:]
        elif wraw in {"05", "5"}:
            w = "0.5"
        elif wraw in {"035", "35"}:
            w = "0.35"
        elif wraw in {"075", "75"}:
            w = "0.75"
        else:
            w = wraw
        return f"w{w}_d{d}_s{img}{gc}"
    if m.lastindex and m.lastindex >= 4 and m.group(1):
        return f"w{m.group(1)}_d{m.group(2)}_s{m.group(3)}{m.group(4)}"
    return None


def norm_tag_key(tag: str) -> str | None:
    m = re.search(r"w0?(\d+)_d(\d{4})_s(\d+)([gc])", tag)
    if not m:
        return None
    wraw, d, img, gc = m.groups()
    mapping = {"035": "0.35", "35": "0.35", "05": "0.5", "5": "0.5", "075": "0.75", "75": "0.75", "1": "1", "10": "1"}
    w = mapping.get(wraw, wraw if "." in wraw else (f"0.{wraw[1:]}" if wraw.startswith("0") else wraw))
    if wraw == "1" or wraw == "10":
        w = "1"
    return f"w{w}_d{d}_s{img}{gc}"


def load_energy(path: Path, fp_backend: str, i8_backend: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            key = norm_tag_key(r.get("tag") or r.get("model") or "")
            if not key:
                continue
            be = (r.get("backend") or "").lower()
            e = float(r["energy_mj_per_infer_net_mean"])
            out.setdefault(key, {})
            if be == fp_backend:
                out[key]["e_fp"] = e
            elif be == i8_backend:
                out[key]["e_i8"] = e
    return out


def load_lat_pair(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            model = r.get("model") or ""
            key = short_key(model)
            if not key:
                continue
            dtype = (r.get("dtype") or "").lower()
            p50 = None
            for k in ("latency_ms_p50", "p50_ms"):
                if r.get(k):
                    p50 = float(r[k])
                    break
            if p50 is None and r.get("latency_us_p50"):
                p50 = float(r["latency_us_p50"]) / 1000.0
            if p50 is None:
                continue
            out.setdefault(key, {})
            if dtype == "fp32":
                out[key]["l_fp"] = p50
            elif dtype == "int8":
                out[key]["l_i8"] = p50
    return out


def load_orin_lat() -> dict[str, dict]:
    """Orin: INT8 from orin_trt_int8_matrix; FP16 from orin_trt_matrix if present."""
    out: dict[str, dict] = {}
    i8 = ROOT / "results/orin_trt_int8_matrix.csv"
    fp = ROOT / "results/orin_trt_matrix.csv"
    for path, field in ((i8, "l_i8"), (fp, "l_fp")):
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                key = short_key(r.get("model") or "")
                if not key:
                    continue
                us = r.get("latency_us_p50")
                if not us:
                    continue
                out.setdefault(key, {})[field] = float(us) / 1000.0
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    series = []
    # H618
    e_h = load_energy(ROOT / "results/power/h618_p5s1/energy_matrix.csv", "ncnn_fp32", "ncnn_int8")
    # Prefer predistill latency (matches energy stems); fall back formal
    l_h = load_lat_pair(ROOT / "results/h618_v4/matrix_summary.csv")
    if not l_h:
        # build from overnight / formal if needed — use flops+formal+predistill txts via formal baked as approx
        l_h = load_lat_pair(ROOT / "results/h618_v4_formal_baked/matrix_summary.csv")
        l_h2 = load_lat_pair(ROOT / "results/h618_v4_flops_ctrl_baked/matrix_summary.csv")
        for k, v in l_h2.items():
            l_h.setdefault(k, {}).update(v)
    # Also parse from energy-aligned predistill board matrices under results/h618_v4
    # Rebuild latency from RESULT txts for predistill stems present in energy
    pre_dir = ROOT / "results/h618_v4"
    if pre_dir.exists():
        for p in list(pre_dir.glob("*_fp32.txt")) + list(pre_dir.glob("*_int8.txt")):
            m = re.match(r"(.+)_(fp32|int8)\.txt$", p.name)
            if not m:
                continue
            key = short_key(m.group(1))
            if not key:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            mm = re.search(r"latency_us_p50=([\d.]+)", text)
            if not mm:
                continue
            l_h.setdefault(key, {})
            if m.group(2) == "fp32":
                l_h[key]["l_fp"] = float(mm.group(1)) / 1000.0
            else:
                l_h[key]["l_i8"] = float(mm.group(1)) / 1000.0

    for key, ev in e_h.items():
        lv = l_h.get(key, {})
        if not all(k in ev for k in ("e_fp", "e_i8")):
            continue
        if not all(k in lv for k in ("l_fp", "l_i8")):
            continue
        series.append(
            {
                "device": "h618",
                "key": key,
                "lat_ratio": lv["l_i8"] / lv["l_fp"],
                "ene_ratio": ev["e_i8"] / ev["e_fp"],
                "ref": "FP32",
            }
        )

    # T527 NCNN
    e_t = load_energy(ROOT / "results/power/t527_p5s1/energy_matrix.csv", "ncnn_fp32", "ncnn_int8")
    l_t: dict[str, dict] = {}
    tdir = ROOT / "results/t527_v4"
    if tdir.exists():
        for p in list(tdir.glob("*_fp32.txt")) + list(tdir.glob("*_int8.txt")):
            m = re.match(r"(.+)_(fp32|int8)\.txt$", p.name)
            if not m:
                continue
            key = short_key(m.group(1))
            if not key:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            mm = re.search(r"latency_us_p50=([\d.]+)", text)
            if not mm:
                continue
            l_t.setdefault(key, {})
            if m.group(2) == "fp32":
                l_t[key]["l_fp"] = float(mm.group(1)) / 1000.0
            else:
                l_t[key]["l_i8"] = float(mm.group(1)) / 1000.0
    for key, ev in e_t.items():
        lv = l_t.get(key, {})
        if not all(k in ev for k in ("e_fp", "e_i8")):
            continue
        if not all(k in lv for k in ("l_fp", "l_i8")):
            continue
        series.append(
            {
                "device": "t527",
                "key": key,
                "lat_ratio": lv["l_i8"] / lv["l_fp"],
                "ene_ratio": ev["e_i8"] / ev["e_fp"],
                "ref": "FP32",
            }
        )

    # Orin INT8 vs FP16
    e_o = load_energy(ROOT / "results/power/orin_p5s1/energy_matrix.csv", "trt_fp16", "trt_int8")
    if not e_o:
        # try alternate backend names
        e_o = load_energy(ROOT / "results/power/orin_p5s1/energy_matrix.csv", "fp16", "int8")
    l_o = load_orin_lat()
    for key, ev in e_o.items():
        lv = l_o.get(key, {})
        if not all(k in ev for k in ("e_fp", "e_i8")):
            continue
        if not all(k in lv for k in ("l_fp", "l_i8")):
            continue
        series.append(
            {
                "device": "orin",
                "key": key,
                "lat_ratio": lv["l_i8"] / lv["l_fp"],
                "ene_ratio": ev["e_i8"] / ev["e_fp"],
                "ref": "FP16",
            }
        )

    with (OUT / "fig8_bitwidth_quadrants.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["device", "key", "lat_ratio", "ene_ratio", "ref"])
        w.writeheader()
        w.writerows(series)

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from paper_style import H618, ORIN, T527, TEXT_IN, apply, save

    style = {
        "h618": {"c": H618, "m": "D", "label": "H618 INT8/FP32"},
        "t527": {"c": T527, "m": "s", "label": "T527 INT8/FP32"},
        "orin": {"c": ORIN, "m": "o", "label": "Orin INT8/FP16"},
    }
    apply()
    fig, ax = plt.subplots(figsize=(TEXT_IN, 3.35), layout="constrained")
    ax.axhline(1.0, color="0.35", lw=0.7, zorder=1)
    ax.axvline(1.0, color="0.35", lw=0.7, zorder=1)
    for device, st in style.items():
        pts = [p for p in series if p["device"] == device]
        if not pts:
            continue
        ax.scatter(
            [p["lat_ratio"] for p in pts],
            [p["ene_ratio"] for p in pts],
            c=st["c"], marker=st["m"], s=28,
            label=f"{st['label']} (n={len(pts)})", zorder=3,
        )
    ax.set_xlabel("Latency ratio (INT8 / baseline)")
    ax.set_ylabel("Energy ratio (INT8 / baseline)")
    xmax = max(1.4, max(p["lat_ratio"] for p in series) * 1.08)
    ymax = max(1.4, max(p["ene_ratio"] for p in series) * 1.08)
    ax.set_xlim(0.55, xmax)
    ax.set_ylim(0.55, ymax)
    ax.text(0.72, 0.68, "faster\nand cheaper", fontsize=7, color="0.35", ha="center")
    ax.text(1.22, 1.22, "slower\nand costlier", fontsize=7, color="0.35", ha="center")
    ax.text(1.35, 0.60, "slower,\ncheaper", fontsize=7, color="0.35", ha="center")
    ax.text(0.72, 1.28, "faster,\ncostlier", fontsize=7, color="0.35", ha="center")
    fig.legend(frameon=False, loc="outside lower center", ncol=3)
    save(fig, "fig8_bitwidth_quadrants")
    plt.close(fig)
    print(f"DONE n={len(series)}")
    print({d: sum(1 for p in series if p["device"] == d) for d in style})
    return 0 if series else 1


if __name__ == "__main__":
    raise SystemExit(main())
