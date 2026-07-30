#!/usr/bin/env python3
"""MCU LUT fidelity vs e2e (P4/S3 anchors). Uses conv+dw+gemm; affine optional."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lut"))
sys.path.insert(0, str(ROOT))
from topo_ops import expand_subnet_ops, op_key  # noqa: E402


def load_lut(path: Path) -> dict[tuple, float]:
    lut: dict[tuple, float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok")).lower() not in {"1", "true"}:
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
            lut[op_key(o)] = float(o["p50_us"])
    return lut


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut", type=Path, default=ROOT / "lut/ops_espdl_v4_firstpass.csv")
    ap.add_argument("--e2e", type=Path, default=ROOT / "lut/e2e_v4_firstpass.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "lut/fidelity_mcu_v4.json")
    ap.add_argument("--fit-affine", action="store_true")
    ap.add_argument("--fit-per-device", action="store_true")
    args = ap.parse_args()

    lut = load_lut(args.lut)
    e2e = [
        r
        for r in csv.DictReader(args.e2e.open(encoding="utf-8"))
        if r["device"] in {"esp32p4", "esp32s3"}
    ]
    rows = []
    for r in e2e:
        be = r["device"]
        ops = expand_subnet_ops(
            width=float(r["width"]),
            depths=tuple(int(x) for x in r["depths"].replace('"', "").split(",")),
            img_size=int(r["img_size"]),
            dtype=r["dtype"],
            backend=be,
        )
        total = 0.0
        miss = []
        for o in ops:
            v = lut.get(op_key(o))
            if v is None:
                if o["op"] in {"add", "gap"}:
                    cin = int(o.get("cin") or 1)
                    hh = int(o.get("h") or 1)
                    ww = int(o.get("w") or hh)
                    v = max(1.0, cin * hh * ww / 1e5)
                else:
                    miss.append(o.get("note") or o["op"])
                    v = 0.0
            total += v
        rows.append(
            {
                "device": be,
                "measured_us": float(r["mean_latency_us"]),
                "sum_ops_us": total,
                "pred_us": total,
                "n_ops": len(ops),
                "n_missing": len(miss),
                "missing": miss[:12],
            }
        )

    def fit_ab(group: list[dict]) -> tuple[float, float]:
        n = len(group)
        if n == 1 and group[0]["sum_ops_us"] > 0:
            return group[0]["measured_us"] / group[0]["sum_ops_us"], 0.0
        sum_s = sum(x["sum_ops_us"] for x in group)
        sum_m = sum(x["measured_us"] for x in group)
        sum_s2 = sum(x["sum_ops_us"] ** 2 for x in group)
        sum_ms = sum(x["measured_us"] * x["sum_ops_us"] for x in group)
        den = n * sum_s2 - sum_s * sum_s
        a = (n * sum_ms - sum_m * sum_s) / den if den else 1.0
        b = (sum_m - a * sum_s) / n
        return a, b

    fit_meta: dict = {}
    if args.fit_per_device:
        by_dev: dict[str, list] = {}
        for r in rows:
            by_dev.setdefault(r["device"], []).append(r)
        for dev, group in by_dev.items():
            a, b = fit_ab(group)
            fit_meta[dev] = {"a": a, "b": b, "n": len(group)}
            for r in group:
                r["pred_us"] = a * r["sum_ops_us"] + b
                r["fit_scale"] = a
                r["fit_bias"] = b
        print("fit_per_device:", fit_meta)
    elif args.fit_affine and len(rows) >= 2:
        a, b = fit_ab(rows)
        fit_meta = {"global": {"a": a, "b": b, "n": len(rows)}}
        for r in rows:
            r["pred_us"] = a * r["sum_ops_us"] + b
            r["fit_scale"] = a
            r["fit_bias"] = b
        print(f"fit_affine a={a:.4f} b={b:.1f}")

    xs = [r["measured_us"] for r in rows]
    ys = [r["pred_us"] for r in rows]
    mape = sum(abs(y - x) / max(x, 1e-9) for x, y in zip(xs, ys)) / len(xs) * 100 if xs else float("nan")
    rcoef = pearson(xs, ys)
    n_cfg = len({(r["device"], round(r["measured_us"])) for r in rows})
    report = {
        "n": len(rows),
        "pearson_r": rcoef,
        "mape_pct": mape,
        "pass_mape": bool(mape <= 10.0),
        "pass_r": bool(rcoef >= 0.95) if len(rows) >= 3 and not math.isnan(rcoef) else False,
        "pass": bool(mape <= 10.0 and len(rows) >= 4 and (not math.isnan(rcoef) and rcoef >= 0.95)),
        "fit": fit_meta,
        "note": (
            f"{len(rows)} MCU e2e points (d2222+d3333 × P4/S3); "
            "gemm mined as 1x1 conv; add/gap proxied; "
            "prefer --fit-affine (global) or --fit-per-device (≥2 pts/device)"
        ),
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        print(
            f"{r['device']}: meas={r['measured_us']:.0f} sum={r['sum_ops_us']:.0f} "
            f"pred={r['pred_us']:.0f} miss={r['n_missing']} "
            f"err%={abs(r['pred_us']-r['measured_us'])/r['measured_us']*100:.1f}"
        )
    print(f"r={rcoef:.4f} MAPE={mape:.2f}% PASS={report['pass']} -> {args.out}")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
