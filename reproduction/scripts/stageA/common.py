#!/usr/bin/env python3
"""Stage-A shared helpers: subnet parsing, proxy metrics, measured-latency pool, operator LUTs.

All Stage-A scripts write under ``results/stageA/``.
"""
from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "stageA"
OUT.mkdir(parents=True, exist_ok=True)
for p in (ROOT, ROOT / "lut"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from topo_ops import expand_subnet_ops  # noqa: E402

WIDTHS = (0.35, 0.5, 0.75, 1.0)
DEPTHS = (2, 3, 4)
RES = (64, 96, 112)
STEM, HEAD_MID, NUM_CLASSES = 32, 1280, 8
STAGES = ((48, 2, 4), (72, 2, 4), (144, 2, 6), (288, 2, 6))

_KEY_RE = re.compile(r"w(\d*\.?\d+)_d(\d)(\d)(\d)(\d)_s(\d+)([gc])?")


def make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def norm_width(tok: str) -> float:
    """'035'->0.35, '05'->0.5, '075'->0.75, '1'->1.0, '0.5'->0.5."""
    if "." in tok:
        return float(tok)
    if tok.startswith("0") and len(tok) > 1:
        return float("0." + tok[1:])
    return float(tok)


def arch_key(width: float, depths, img: int) -> str:
    d = "".join(str(int(x)) for x in depths)
    w = f"{width:g}"
    return f"w{w}_d{d}_s{img}{'g' if img == 64 else 'c'}"


def parse_key(s: str) -> tuple[float, tuple[int, int, int, int], int] | None:
    m = _KEY_RE.search(s)
    if not m:
        return None
    w = norm_width(m.group(1))
    d = tuple(int(m.group(i)) for i in range(2, 6))
    img = int(m.group(6))
    return w, d, img  # type: ignore[return-value]


def all_subnets() -> list[tuple[float, tuple[int, int, int, int], int]]:
    out = []
    for w in WIDTHS:
        for d1 in DEPTHS:
            for d2 in DEPTHS:
                for d3 in DEPTHS:
                    for d4 in DEPTHS:
                        for r in RES:
                            out.append((w, (d1, d2, d3, d4), r))
    return out


def flops_search_proxy(width: float, depths, img: int) -> float:
    """Exact replica of ElasticFERSuperNet.estimate_flops (v3_wide, slimmable) used by the FLOPs arm."""
    stem_c = make_divisible(STEM * width)
    h = img // 2
    macs = 3 * stem_c * 9 * h * h + stem_c * stem_c * 9 * h * h
    in_ch = stem_c
    for si, (out_max, stride, expand) in enumerate(STAGES):
        out_ch = make_divisible(out_max * width)
        for bi in range(depths[si]):
            s = stride if bi == 0 else 1
            hid = make_divisible(in_ch * expand)
            oh = h // s
            macs += in_ch * hid * oh * oh + hid * 9 * oh * oh + hid * out_ch * oh * oh
            h = oh
            in_ch = out_ch
    mid = make_divisible(HEAD_MID * width)
    macs += in_ch * mid * h * h + mid * NUM_CLASSES
    return float(macs)


def subnet_ops(width: float, depths, img: int, dtype: str = "fp32", backend: str = "h618_ncnn") -> list[dict]:
    return expand_subnet_ops(width=width, depths=tuple(depths), img_size=img, dtype=dtype, backend=backend)


def op_out_hw(o: dict) -> int:
    h = int(o.get("h") or 1)
    s = int(o.get("s") or 1)
    return h // s


def subnet_features(width: float, depths, img: int) -> dict:
    """Proxy metrics + operator-level features for one subnet."""
    ops = subnet_ops(width, depths, img)
    f = {
        "width": width, "img": img, "gray": int(img == 64),
        "d1": depths[0], "d2": depths[1], "d3": depths[2], "d4": depths[3],
        "depth_blocks": int(sum(depths)),
    }
    macs = {"stem": 0.0, "pw": 0.0, "dw": 0.0, "se": 0.0, "head": 0.0, "fc": 0.0}
    stage_macs = [0.0, 0.0, 0.0, 0.0]
    params = 0.0
    act = 0.0
    n = {"conv": 0, "dwconv": 0, "gemm": 0, "add": 0, "gap": 0}
    for o in ops:
        op = o["op"]
        n[op] = n.get(op, 0) + 1
        note = str(o.get("note", ""))
        sidx = None
        m = re.search(r"_s(\d)_b", note)
        if m:
            sidx = int(m.group(1))
        if op in ("conv", "dwconv"):
            cin, cout, k = int(o["cin"]), int(o["cout"]), int(o["k"])
            g = int(o.get("groups") or 1)
            oh = op_out_hw(o)
            mac = (cin // g) * cout * k * k * oh * oh
            params += (cin // g) * cout * k * k + 2 * cout
            act += cout * oh * oh
            if note.startswith("stem"):
                macs["stem"] += mac
            elif note == "head_pw":
                macs["head"] += mac
            elif op == "dwconv":
                macs["dw"] += mac
            else:
                macs["pw"] += mac
        elif op == "gemm":
            kk, nn = int(o["k"]), int(o["n"])
            mac = kk * nn
            params += kk * nn + nn
            if note == "fc":
                macs["fc"] += mac
            else:
                macs["se"] += mac
        else:
            mac = 0.0
        if sidx is not None:
            stage_macs[sidx] += mac
    total = sum(macs.values())
    f.update({f"mac_{k}": v for k, v in macs.items()})
    f.update({f"mac_stage{i+1}": v for i, v in enumerate(stage_macs)})
    f.update({f"n_{k}": v for k, v in n.items()})
    f["n_layers"] = n["conv"] + n["dwconv"] + n["gemm"]
    f["macs_full"] = total
    f["flops_search"] = flops_search_proxy(width, depths, img)
    f["params"] = params
    f["act_elems"] = act
    f["dw_share"] = macs["dw"] / total if total else 0.0
    return f


def features_frame(keys: list[str]) -> pd.DataFrame:
    rows = []
    for k in keys:
        w, d, r = parse_key(k)  # type: ignore[misc]
        rows.append({"key": k, **subnet_features(w, d, r)})
    return pd.DataFrame(rows).set_index("key")


# ---------------------------------------------------------------- measured pool
NCNN_FILES = {
    "h618": [
        "results/h618_v4/matrix_summary.csv",
        "results/h618_v4_flops_ctrl_baked/matrix_summary.csv",
        "results/h618_v4_formal_baked/matrix_summary.csv",
        "results/h618_v4_formal_baked_expw/matrix_summary.csv",
    ],
    "t527": [
        "results/t527_v4/matrix_summary.csv",
        "results/t527_v4_flops_ctrl_baked/matrix_summary.csv",
        "results/t527_v4_formal_baked/matrix_summary.csv",
        "results/t527_v4_formal_baked_expw/matrix_summary.csv",
    ],
}


def _p50_ms(r: dict) -> float:
    for c in ("latency_us_p50",):
        if r.get(c) not in (None, "", "nan"):
            return float(r[c]) / 1000.0
    for c in ("latency_ms_p50", "p50_ms"):
        if r.get(c) not in (None, "", "nan"):
            return float(r[c])
    raise KeyError(r)


def measured_rows() -> pd.DataFrame:
    """Every measured e2e row (before de-duplication): device, backend_dtype, key, lat_ms, source."""
    rows = []
    for dev, files in NCNN_FILES.items():
        for f in files:
            with (ROOT / f).open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    pk = parse_key(r["model"])
                    rows.append({
                        "device": dev, "platform": f"{dev.upper()} NCNN {r['dtype'].upper()}",
                        "dtype": r["dtype"], "key": arch_key(*pk), "lat_ms": _p50_ms(r),
                        "dataset": "expw" if "_expw_" in r["model"] else "ferplus", "source": f,
                    })
    # B10/O1: 191 supernet-sliced architectures in one protocol; supersedes the 6 + 13 distilled-weight
    # Orin points below. Each Orin bench process lands in a fast or a slow level, so the per-engine
    # minimum over 10 processes (o1_min_latency.csv) is used when present, else the first O1 round.
    o1 = ROOT / "results/stageB/orin_sweep/o1_latency.csv"
    o1m = ROOT / "results/stageB/orin_sweep/o1_min_latency.csv"
    if o1m.exists() or o1.exists():
        src = o1m if o1m.exists() else o1
        for r in csv.DictReader(src.open(encoding="utf-8")):
            if r.get("round", "0") != "0":
                continue
            pk = parse_key(r["arch"])
            dt = r["precision"]
            rows.append({"device": "orin", "platform": f"Orin TRT {dt.upper()}", "dtype": dt,
                         "key": arch_key(*pk), "lat_ms": float(r["latency_us_p50"]) / 1000.0,
                         "dataset": "ferplus", "source": str(src.relative_to(ROOT)).replace("\\", "/")})
    else:
        for f, dt in (("results/orin_trt_matrix.csv", "fp16"), ("results/orin_trt_int8_matrix.csv", "int8")):
            for r in csv.DictReader((ROOT / f).open(encoding="utf-8")):
                pk = parse_key(r["model"])
                rows.append({"device": "orin", "platform": f"Orin TRT {dt.upper()}", "dtype": dt,
                             "key": arch_key(*pk), "lat_ms": float(r["latency_us_p50"]) / 1000.0,
                             "dataset": "ferplus", "source": f})
    # Stage B0: 13 architectures re-benched on the same Orin (1000 iters each).
    ext = ROOT / "results/stageB/orin_trt_extended.csv"
    if ext.exists() and not o1.exists():
        for r in csv.DictReader(ext.open(encoding="utf-8")):
            pk = parse_key(r["arch"])
            dt = r["precision"]
            rows.append({"device": "orin", "platform": f"Orin TRT {dt.upper()}", "dtype": dt,
                         "key": arch_key(*pk), "lat_ms": float(r["latency_us_p50"]) / 1000.0,
                         "dataset": "ferplus", "source": "results/stageB/orin_trt_extended.csv"})
    for r in csv.DictReader((ROOT / "results/t527_v4/npu_matrix_summary.csv").open(encoding="utf-8")):
        pk = parse_key(r["model"])
        rows.append({"device": "t527npu", "platform": "T527 NPU UINT8", "dtype": "uint8",
                     "key": arch_key(*pk), "lat_ms": float(r["latency_us_p50"]) / 1000.0,
                     "dataset": "ferplus", "source": "results/t527_v4/npu_matrix_summary.csv"})
    for r in csv.DictReader((ROOT / "lut/e2e_mcu_truth.csv").open(encoding="utf-8")):
        dev = r["device"]
        pk = parse_key(r["model"])
        rows.append({"device": dev, "platform": f"{'ESP32-P4' if dev == 'esp32p4' else 'ESP32-S3'} ESP-DL INT8",
                     "dtype": "int8", "key": arch_key(*pk), "lat_ms": float(r["mean_latency_ms"]),
                     "dataset": "ferplus", "source": "lut/e2e_mcu_truth.csv"})
    return pd.DataFrame(rows)


PLATFORM_ORDER = [
    "H618 NCNN FP32", "H618 NCNN INT8", "H618 NCNN AUTO",
    "T527 NCNN FP32", "T527 NCNN INT8", "T527 NCNN AUTO",
    "T527 NPU UINT8", "Orin TRT FP16", "Orin TRT INT8",
    "ESP32-P4 ESP-DL INT8", "ESP32-S3 ESP-DL INT8",
]


def measured_pool() -> pd.DataFrame:
    """De-duplicated pool: one latency per (platform, arch) = median over repeated artifacts.

    Adds an AUTO platform for NCNN boards: min(FP32, INT8) per arch (the dtype decision the LUT search makes).
    """
    raw = measured_rows()
    g = raw.groupby(["platform", "device", "dtype", "key"]).lat_ms.agg(["median", "count", "min", "max"]).reset_index()
    g = g.rename(columns={"median": "lat_ms", "count": "n_rep"})
    autos = []
    for dev in ("h618", "t527"):
        sub = g[g.device == dev]
        piv = sub.pivot_table(index="key", columns="dtype", values="lat_ms")
        piv = piv.dropna()
        for k, r in piv.iterrows():
            autos.append({"platform": f"{dev.upper()} NCNN AUTO", "device": dev, "dtype": "auto", "key": k,
                          "lat_ms": float(min(r["fp32"], r["int8"])), "n_rep": 1,
                          "min": float(min(r["fp32"], r["int8"])), "max": float(min(r["fp32"], r["int8"]))})
    return pd.concat([g, pd.DataFrame(autos)], ignore_index=True)


# ---------------------------------------------------------------- operator LUTs
def _num(v):
    if v in (None, "") or (isinstance(v, float) and math.isnan(v)):
        return None
    f = float(v)
    return int(f) if f.is_integer() else f


def _okey(op, cin, cout, k, s, h, w, n, groups, dtype):
    return (op, _num(cin), _num(cout), _num(k), _num(s), _num(h), _num(w), _num(n), _num(groups), dtype)


def load_op_lut(device: str) -> dict:
    lut = {}
    if device == "h618":
        df = pd.read_csv(ROOT / "lut/ops_h618_v4_full.csv")
        col = "p50_us"
        df = df[df.ok.astype(str).str.lower().isin(["true", "1"])]
    elif device == "t527":
        df = pd.read_csv(ROOT / "lut/ops_t527_v4_full.csv")
        col = "latency_us_p50"
        df = df[df.ok.astype(str).isin(["1", "True", "true"])]
    else:
        raise ValueError(device)
    for _, r in df.iterrows():
        lut[_okey(r["op"], r["cin"], r["cout"], r["k"], r["s"], r["h"], r["w"], r["n"], r["groups"],
                  r["dtype"])] = float(r[col])
    return lut


def lut_sum_us(lut: dict, width: float, depths, img: int, dtype: str) -> tuple[float, int]:
    """Sum of operator latencies; H618 add ops use the same proxy as lut/predict_e2e.py. Returns (sum, n_missing)."""
    total, miss = 0.0, 0
    for o in subnet_ops(width, depths, img, dtype=dtype):
        k = _okey(o["op"], o.get("cin"), o.get("cout"), o.get("k"), o.get("s"), o.get("h"), o.get("w"),
                  o.get("n"), o.get("groups"), dtype)
        v = lut.get(k)
        if v is None and dtype == "int8":
            v = lut.get(k[:-1] + ("fp32",))
        if v is None:
            if o["op"] in ("add", "gap"):
                cin = int(o.get("cin") or 1)
                hh = int(o.get("h") or 1)
                ww = int(o.get("w") or hh)
                v = max(1.0, cin * hh * ww / 1e5)
            else:
                miss += 1
                v = 0.0
        total += v
    return total, miss


# ---------------------------------------------------------------- stats
def bootstrap_ci(x, stat=np.median, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = np.apply_along_axis(stat, 1, x[idx])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def rank_biserial(d) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014) on non-zero differences."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.abs(d))
    rp, rn = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rn) / r.sum())


def fmt_ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"
