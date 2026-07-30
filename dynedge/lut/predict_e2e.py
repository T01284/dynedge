#!/usr/bin/env python3
"""LUT fidelity: sum op latencies vs e2e measured rows.

Target (Phase 1): Pearson r >= 0.95, MAPE <= 10% on H618 e2e rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lut"))

from topo_ops import expand_subnet_ops, op_key  # noqa: E402


def parse_depths(s: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in str(s).replace('"', "").split(",")]
    if len(parts) != 4:
        raise ValueError(s)
    return parts[0], parts[1], parts[2], parts[3]


def load_lut(path: Path) -> dict[tuple, float]:
    lut: dict[tuple, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok", "")).lower() not in {"1", "true", "True"}:
                continue
            if not r.get("p50_us"):
                continue
            o = dict(r)
            for num in ("cin", "cout", "k", "s", "h", "w", "m", "n", "groups", "width"):
                if o.get(num) in (None, ""):
                    o[num] = None
                else:
                    try:
                        o[num] = float(o[num]) if "." in str(o[num]) else int(float(o[num]))
                    except ValueError:
                        o[num] = None
            o["backend"] = "h618_ncnn"
            lut[op_key(o)] = float(o["p50_us"])
    return lut


def lookup(lut: dict[tuple, float], o: dict) -> float | None:
    o = {**o, "backend": "h618_ncnn"}
    k = op_key(o)
    if k in lut:
        return lut[k]
    # try without width in note-only — already in key without width
    # soft: ignore note; key already ignores note
    return None


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", type=Path, default=ROOT / "lut/e2e_v4_firstpass.csv")
    ap.add_argument("--lut", type=Path, default=ROOT / "lut/ops_h618_v4_full.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "lut/fidelity_h618_v4.json")
    ap.add_argument("--device", default="h618")
    ap.add_argument("--overhead-us", type=float, default=0.0, help="constant framework overhead added to sum")
    ap.add_argument("--scale", type=float, default=1.0, help="global scale on LUT sum (fit optional)")
    ap.add_argument("--fit-scale", action="store_true", help="least-squares scale pred≈a*sum")
    ap.add_argument("--fit-affine", action="store_true", help="least-squares pred≈a*sum+b")
    ap.add_argument("--fit-per-dtype", action="store_true", help="separate a[,b] per dtype")
    args = ap.parse_args()

    lut = load_lut(args.lut)
    print(f"lut entries={len(lut)}")

    rows = []
    with args.e2e.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("device") != args.device:
                continue
            width = float(r["width"])
            depths = parse_depths(r["depths"])
            img = int(r["img_size"])
            dtype = r["dtype"]
            measured = float(r["mean_latency_us"])
            ops = expand_subnet_ops(
                width=width,
                depths=depths,
                img_size=img,
                dtype=dtype,
                backend="h618_ncnn",
            )
            missing = []
            total = 0.0
            for o in ops:
                # skip add/gap if not in lut — assign small proxy
                v = lookup(lut, o)
                if v is None:
                    if o["op"] in {"add", "gap"}:
                        # tiny proxy: 1us * spatial * channels / 1e5
                        cin = int(o.get("cin") or 1)
                        hh = int(o.get("h") or 1)
                        ww = int(o.get("w") or hh)
                        v = max(1.0, cin * hh * ww / 1e5)
                    else:
                        missing.append(o.get("note") or o["op"])
                        v = 0.0
                total += v
            pred = args.scale * total + args.overhead_us
            rows.append(
                {
                    "model": r["model"],
                    "dtype": dtype,
                    "width": width,
                    "depths": list(depths),
                    "img_size": img,
                    "measured_us": measured,
                    "pred_us": pred,
                    "sum_ops_us": total,
                    "n_ops": len(ops),
                    "n_missing": len(missing),
                    "missing_head": missing[:8],
                }
            )

    if args.fit_per_dtype and rows:
        from collections import defaultdict

        by = defaultdict(list)
        for r in rows:
            if r["n_missing"] == 0:
                by[r["dtype"]].append(r)
        scales = {}
        for dtype, rs in by.items():
            if args.fit_affine and len(rs) >= 2:
                # solve [s,1; ...] [a,b] = m
                # a = (n*sum(ms)-sum(m)sum(s))/(n*sum(s2)-sum(s)^2)
                n = len(rs)
                sum_s = sum(x["sum_ops_us"] for x in rs)
                sum_m = sum(x["measured_us"] for x in rs)
                sum_s2 = sum(x["sum_ops_us"] ** 2 for x in rs)
                sum_ms = sum(x["measured_us"] * x["sum_ops_us"] for x in rs)
                den = n * sum_s2 - sum_s * sum_s
                a = (n * sum_ms - sum_m * sum_s) / den if den else 1.0
                b = (sum_m - a * sum_s) / n
            else:
                num = sum(x["measured_us"] * x["sum_ops_us"] for x in rs)
                den = sum(x["sum_ops_us"] ** 2 for x in rs)
                a = num / den if den else 1.0
                b = args.overhead_us
            scales[dtype] = (a, b)
            for r in rs:
                r["pred_us"] = a * r["sum_ops_us"] + b
                r["fit_scale"] = a
                r["fit_bias"] = b
        # also apply to any leftover
        for r in rows:
            if "fit_scale" not in r and r["dtype"] in scales:
                a, b = scales[r["dtype"]]
                r["pred_us"] = a * r["sum_ops_us"] + b
        print(f"fit_per_dtype={scales}")
    elif args.fit_affine and rows:
        rs = [r for r in rows if r["n_missing"] == 0]
        n = len(rs)
        sum_s = sum(x["sum_ops_us"] for x in rs)
        sum_m = sum(x["measured_us"] for x in rs)
        sum_s2 = sum(x["sum_ops_us"] ** 2 for x in rs)
        sum_ms = sum(x["measured_us"] * x["sum_ops_us"] for x in rs)
        den = n * sum_s2 - sum_s * sum_s
        a = (n * sum_ms - sum_m * sum_s) / den if den else 1.0
        b = (sum_m - a * sum_s) / n if n else 0.0
        for r in rows:
            r["pred_us"] = a * r["sum_ops_us"] + b
            r["fit_scale"] = a
            r["fit_bias"] = b
        print(f"fit_affine a={a:.4f} b={b:.1f}")
    elif args.fit_scale and rows:
        # fit pred = a * sum_ops  ≈ measured  => a = sum(m*s)/sum(s^2)
        num = sum(r["measured_us"] * r["sum_ops_us"] for r in rows if r["n_missing"] == 0)
        den = sum(r["sum_ops_us"] ** 2 for r in rows if r["n_missing"] == 0)
        a = num / den if den else 1.0
        for r in rows:
            r["pred_us"] = a * r["sum_ops_us"] + args.overhead_us
            r["fit_scale"] = a
        print(f"fit_scale={a:.4f}")

    usable = [r for r in rows if r["n_missing"] == 0]
    xs = [r["measured_us"] for r in usable]
    ys = [r["pred_us"] for r in usable]
    mape = (
        sum(abs(y - x) / max(x, 1e-9) for x, y in zip(xs, ys)) / len(xs) * 100.0
        if xs
        else float("nan")
    )
    r = pearson(xs, ys)
    report = {
        "n_e2e": len(rows),
        "n_usable": len(usable),
        "n_missing_any": sum(1 for x in rows if x["n_missing"] > 0),
        "pearson_r": r,
        "mape_pct": mape,
        "pass_r": bool(r >= 0.95) if xs else False,
        "pass_mape": bool(mape <= 10.0) if xs else False,
        "pass": bool(xs) and r >= 0.95 and mape <= 10.0,
        "lut_path": str(args.lut),
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"usable={len(usable)}/{len(rows)}  r={r:.4f}  MAPE={mape:.2f}%  "
        f"PASS={report['pass']} (r>=0.95 & MAPE<=10%)"
    )
    print(f"wrote {args.out}")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
