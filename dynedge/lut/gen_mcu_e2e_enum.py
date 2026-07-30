#!/usr/bin/env python3
"""MCU exhaustive e2e LUT — candidate space + truth-table helpers.

Method decision (2026-07-21): MCU abandons op-level latency modeling.
Candidate space is small (@64 gray × width{0.35,0.5} × curated depths ≈24),
so each subnet is fully flashed and timed. LUT = ground-truth table (100% fidelity).

Paper line: "MCU 档候选空间足够小,采用穷举式真机剖析替代算子级建模"
§4.5 also cites: isolated ops are only ~27–52% of wall-clock on P4/S3
(framework overhead dominates) — another nail in FLOPs / even op-sum proxies.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Flash-safe MCU deploy: w0.75@64 already exceeds 4MB factory partition.
MCU_WIDTHS = (0.35, 0.5)
MCU_IMG = 64
MCU_GRAY = True
MCU_DTYPE = "int8"
MCU_DEVICES = ("esp32p4", "esp32s3")

# Flash-safe curated depths (4MB factory partition).
# Empirically: models with many stage-4 blocks or w0.5+depth≥4 exceed app image.
# Full {2,3,4}^4 × widths is NOT deployable; this is the exhaustive *feasible* set.
MCU_DEPTH_PATTERNS_W035: tuple[tuple[int, int, int, int], ...] = (
    (2, 2, 2, 2),
    (2, 2, 2, 3),
    (2, 2, 3, 3),
    (2, 3, 3, 3),
    (3, 3, 3, 3),
    (2, 2, 3, 4),
    (2, 3, 3, 4),
    (2, 3, 4, 4),
    (3, 3, 3, 4),
    (3, 3, 4, 4),
    (3, 4, 4, 4),
    (4, 4, 4, 4),
)
# w0.5: only shallow/mid — deeper exceeds 4MB on both chips
MCU_DEPTH_PATTERNS_W05: tuple[tuple[int, int, int, int], ...] = (
    (2, 2, 2, 2),
    (2, 2, 2, 3),
    (2, 2, 3, 3),
    (2, 3, 3, 3),
    (3, 3, 3, 3),
)

# Back-compat alias used by older callers
MCU_DEPTH_PATTERNS = MCU_DEPTH_PATTERNS_W035

# From w0.35 d2222 anchors: Σops / e2e ≈ 5.4/19.7 (P4), 30.5/58.8 (S3)
OP_WALLCLOCK_SHARE = {
    "esp32p4": 5354.0 / 19699.75,  # ≈0.272
    "esp32s3": 30482.5 / 58824.625,  # ≈0.518
}


def wtag(w: float) -> str:
    if float(w) == int(w):
        return str(int(w))
    return str(w).rstrip("0").rstrip(".")


def depths_str(d: tuple[int, ...]) -> str:
    return ",".join(map(str, d))


def cfg_key(width: float, depths: tuple[int, ...], device: str | None = None) -> str:
    base = f"w{wtag(width)}_d{''.join(map(str, depths))}_s{MCU_IMG}g"
    return f"{device}:{base}" if device else base


def model_stem(width: float, depths: tuple[int, ...], dataset: str = "ferplus") -> str:
    return f"v4_w{wtag(width)}_d{''.join(map(str, depths))}_s{MCU_IMG}g_{dataset}"


def enumerate_cfgs(dataset: str = "ferplus") -> list[dict]:
    rows = []
    for w, patterns in (
        (0.35, MCU_DEPTH_PATTERNS_W035),
        (0.5, MCU_DEPTH_PATTERNS_W05),
    ):
        for depths in patterns:
            rows.append(
                {
                    "dataset": dataset,
                    "width": w,
                    "depths": depths_str(depths),
                    "depths_tuple": depths,
                    "img_size": MCU_IMG,
                    "gray": 1,
                    "dtype": MCU_DTYPE,
                    "key": cfg_key(w, depths),
                    "stem": model_stem(w, depths, dataset),
                    "cli": f"{dataset}:{w}:{depths_str(depths)}:{MCU_IMG}",
                }
            )
    return rows


def load_truth(path: Path) -> dict[tuple, float]:
    """Map (device, width, depths_str) -> mean_latency_us."""
    out: dict[tuple, float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("device") not in MCU_DEVICES:
                continue
            if not r.get("mean_latency_us"):
                continue
            w = float(r["width"])
            depths = r["depths"].replace('"', "")
            out[(r["device"], w, depths)] = float(r["mean_latency_us"])
    return out


def coverage_report(truth: dict[tuple, float], dataset: str = "ferplus") -> dict:
    cfgs = enumerate_cfgs(dataset)
    missing = []
    have = []
    for c in cfgs:
        for dev in MCU_DEVICES:
            k = (dev, c["width"], c["depths"])
            if k in truth:
                have.append({**c, "device": dev, "mean_latency_us": truth[k]})
            else:
                missing.append({**c, "device": dev, "cli": c["cli"]})
    return {
        "n_cfg": len(cfgs),
        "n_device_cfg": len(cfgs) * len(MCU_DEVICES),
        "n_have": len(have),
        "n_missing": len(missing),
        "pct": 100.0 * len(have) / max(1, len(cfgs) * len(MCU_DEVICES)),
        "have": have,
        "missing": missing,
        "op_wallclock_share": {k: round(v, 4) for k, v in OP_WALLCLOCK_SHARE.items()},
        "method": "exhaustive_e2e_truth_table",
        "note": (
            "MCU LUT is ground-truth e2e; no affine fit. "
            "Op-level ESP-DL mining retained only as §4.5 evidence "
            f"(ops≈{100*OP_WALLCLOCK_SHARE['esp32p4']:.0f}–"
            f"{100*OP_WALLCLOCK_SHARE['esp32s3']:.0f}% of wall-clock)."
        ),
    }


def seed_truth_from_e2e(
    e2e_path: Path, out_path: Path
) -> int:
    """Copy MCU rows from e2e_v4_firstpass into dedicated truth CSV."""
    fields = [
        "device",
        "backend",
        "model",
        "width",
        "depths",
        "img_size",
        "gray",
        "dtype",
        "mean_latency_us",
        "mean_latency_ms",
        "source",
    ]
    rows = []
    if e2e_path.exists():
        with e2e_path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("device") not in MCU_DEVICES:
                    continue
                if int(float(r.get("img_size") or 0)) != MCU_IMG:
                    continue
                rows.append(
                    {
                        "device": r["device"],
                        "backend": r.get("backend", "esp-dl"),
                        "model": r.get("model", ""),
                        "width": r["width"],
                        "depths": r["depths"].replace('"', ""),
                        "img_size": MCU_IMG,
                        "gray": r.get("gray", "1"),
                        "dtype": r.get("dtype", "int8"),
                        "mean_latency_us": r["mean_latency_us"],
                        "mean_latency_ms": r.get("mean_latency_ms")
                        or round(float(r["mean_latency_us"]) / 1000.0, 3),
                        "source": r.get("source", ""),
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-enum", type=Path, default=ROOT / "lut/enum_mcu_e2e.json")
    ap.add_argument("--truth", type=Path, default=ROOT / "lut/e2e_mcu_truth.csv")
    ap.add_argument("--e2e-src", type=Path, default=ROOT / "lut/e2e_v4_firstpass.csv")
    ap.add_argument("--seed-truth", action="store_true")
    ap.add_argument("--print-missing", action="store_true")
    ap.add_argument("--print-cli", action="store_true", help="print mine_mcu_e2e_batch --cfgs args")
    args = ap.parse_args()

    cfgs = enumerate_cfgs()
    payload = {
        "method": "exhaustive_e2e",
        "widths": list(MCU_WIDTHS),
        "img_size": MCU_IMG,
        "gray": MCU_GRAY,
        "dtype": MCU_DTYPE,
        "devices": list(MCU_DEVICES),
        "depth_patterns": [list(d) for d in MCU_DEPTH_PATTERNS],
        "n_cfg": len(cfgs),
        "n_e2e_runs": len(cfgs) * len(MCU_DEVICES),
        "op_wallclock_share": OP_WALLCLOCK_SHARE,
        "cfgs": [{k: v for k, v in c.items() if k != "depths_tuple"} for c in cfgs],
    }
    args.out_enum.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out_enum} n_cfg={len(cfgs)} e2e_runs={payload['n_e2e_runs']}")

    if args.seed_truth:
        n = seed_truth_from_e2e(args.e2e_src, args.truth)
        print(f"seeded {args.truth} rows={n}")

    truth = load_truth(args.truth if args.truth.exists() else args.e2e_src)
    rep = coverage_report(truth)
    print(
        f"coverage {rep['n_have']}/{rep['n_device_cfg']} ({rep['pct']:.1f}%) "
        f"op_share P4={100*OP_WALLCLOCK_SHARE['esp32p4']:.1f}% "
        f"S3={100*OP_WALLCLOCK_SHARE['esp32s3']:.1f}%"
    )
    cov_path = ROOT / "lut/mcu_e2e_coverage.json"
    cov_path.write_text(
        json.dumps({k: v for k, v in rep.items() if k not in {"have", "missing"}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    # keep full missing list for runner
    (ROOT / "lut/mcu_e2e_missing.json").write_text(
        json.dumps(rep["missing"], indent=2), encoding="utf-8"
    )

    if args.print_missing:
        for m in rep["missing"]:
            print(f"  MISS {m['device']} {m['key']}")
    if args.print_cli:
        # unique cfgs still missing on any device
        need = sorted({m["cli"] for m in rep["missing"]})
        print("python scripts/mine_mcu_e2e_batch.py --cfgs " + " ".join(need))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
