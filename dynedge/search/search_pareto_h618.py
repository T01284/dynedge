#!/usr/bin/env python3
"""First LUT-based Pareto sketch on H618 (from verified ops LUT + e2e anchors).

Enumerates depth×width×res candidates, predicts latency via LUT sum + affine,
pairs with predistill accuracy from SUMMARY artifacts when available.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lut"))
sys.path.insert(0, str(ROOT))
from topo_ops import expand_subnet_ops, op_key  # noqa: E402

# From H618 fidelity fit
AFFINE = {
    "fp32": (0.8786167764789027, 1187.1673261297146),
    "int8": (0.8055378987527299, 877.7509894135619),
}


def load_lut(path: Path) -> dict:
    lut = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("ok")).lower() not in {"1", "true"}:
                continue
            if r.get("backend") != "h618_ncnn":
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


def predict(lut, width, depths, img_size, dtype) -> tuple[float, int]:
    ops = expand_subnet_ops(
        width=width, depths=depths, img_size=img_size, dtype=dtype, backend="h618_ncnn"
    )
    total = 0.0
    miss = 0
    for o in ops:
        v = lut.get(op_key(o))
        if v is None:
            if o["op"] in {"add", "gap"}:
                cin = int(o.get("cin") or 1)
                hh = int(o.get("h") or 1)
                ww = int(o.get("w") or hh)
                v = max(1.0, cin * hh * ww / 1e5)
            else:
                miss += 1
                v = 0.0
        total += v
    a, b = AFFINE[dtype]
    return a * total + b, miss


def load_acc() -> dict[tuple, float]:
    """Map (dataset,width,depths,img) -> test_acc from predistill json if present."""
    out = {}
    base = ROOT / "results/from_autodl/subnets_v4_predistill"
    if not base.exists():
        return out
    for p in base.glob("subnet_*_v4_predistill.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # flexible field names
        acc = d.get("test_acc") or d.get("test_max") or d.get("best_test") or d.get("acc")
        cfg = d.get("config") or d.get("subnet") or d
        w = cfg.get("width") or d.get("width")
        depths = cfg.get("depths") or d.get("depths")
        img = cfg.get("img_size") or d.get("img_size")
        name = p.name
        ds = "ferplus" if "ferplus" in name else "expw" if "expw" in name else "?"
        if w is None or depths is None:
            continue
        if isinstance(depths, str):
            depths = tuple(int(x) for x in depths.replace("[", "").replace("]", "").split(","))
        else:
            depths = tuple(depths)
        key = (ds, float(w), depths, int(img or 112))
        if acc is not None:
            out[key] = float(acc)
    return out


def pareto_front(rows: list[dict], acc_key="acc", lat_key="pred_ms") -> list[dict]:
    # maximize acc, minimize latency
    cand = [r for r in rows if r.get(acc_key) is not None]
    cand.sort(key=lambda r: (-r[acc_key], r[lat_key]))
    front = []
    best_lat = float("inf")
    for r in cand:
        if r[lat_key] < best_lat:
            front.append(r)
            best_lat = r[lat_key]
    return front


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut", type=Path, default=ROOT / "lut/ops_h618_v4_full.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "lut/pareto_h618_v4_sketch.csv")
    ap.add_argument("--dtype", default="int8")
    args = ap.parse_args()

    lut = load_lut(args.lut)
    acc = load_acc()
    print(f"lut={len(lut)} acc_cfgs={len(acc)}")

    widths = (0.35, 0.5, 0.75, 1.0)
    depth_sets = ((2, 2, 2, 2), (3, 3, 3, 3), (4, 4, 4, 4))
    ress = (64, 96, 112)
    rows = []
    for w in widths:
        for depths in depth_sets:
            for img in ress:
                # gray only at 64 in v4 recipe
                if img == 64 and w not in (0.35,):
                    pass  # still allow prediction
                pred_us, miss = predict(lut, w, depths, img, args.dtype)
                row = {
                    "width": w,
                    "depths": ",".join(map(str, depths)),
                    "img_size": img,
                    "dtype": args.dtype,
                    "pred_us": round(pred_us, 1),
                    "pred_ms": round(pred_us / 1000.0, 3),
                    "n_missing_ops": miss,
                }
                for ds in ("ferplus", "expw"):
                    a = acc.get((ds, w, depths, img))
                    row[f"acc_{ds}"] = a
                # primary acc: prefer ferplus then expw
                row["acc"] = row.get("acc_ferplus") or row.get("acc_expw")
                rows.append(row)

    keys = list(rows[0].keys())
    with args.out.open("w", encoding="utf-8", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=keys)
        wri.writeheader()
        wri.writerows(rows)

    front = pareto_front([r for r in rows if r["acc"] is not None])
    print(f"wrote {args.out} n={len(rows)} pareto_with_acc={len(front)}")
    for r in front[:12]:
        print(
            f"  w={r['width']} d={r['depths']} s={r['img_size']} "
            f"pred={r['pred_ms']:.2f}ms acc={r['acc']}"
        )
    if not front:
        # show cheapest / mid / expensive predictions
        rs = sorted(rows, key=lambda x: x["pred_ms"])
        print("no acc join; latency span:")
        for r in (rs[0], rs[len(rs) // 2], rs[-1]):
            print(f"  w={r['width']} d={r['depths']} s={r['img_size']} pred={r['pred_ms']:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
