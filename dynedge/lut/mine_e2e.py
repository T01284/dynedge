#!/usr/bin/env python3
"""Fill first-pass LUT rows from end-to-end board measurements (MCU / H618).

Op-level mining comes later; this anchors device×model latency for Pareto.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum", type=Path, default=ROOT / "lut/enum_v4_firstpass.json")
    ap.add_argument("--out", type=Path, default=ROOT / "lut/e2e_v4_firstpass.csv")
    ap.add_argument("--board-check", action="append", default=[], help="path=label pairs unused; use --from-json")
    ap.add_argument(
        "--from-json",
        action="append",
        default=[],
        help="board_check.json path (repeatable)",
    )
    args = ap.parse_args()

    rows = []
    for p in args.from_json:
        path = Path(p)
        data = json.loads(path.read_text(encoding="utf-8"))
        # infer device from filename / content
        name = path.name
        if "esp32p4" in name:
            device, backend = "esp32p4", "esp-dl"
        elif "esp32s3" in name:
            device, backend = "esp32s3", "esp-dl"
        else:
            device, backend = "unknown", "unknown"
        rows.append(
            {
                "device": device,
                "backend": backend,
                "model": "v4_w035_d2222_s64g_ferplus",
                "width": 0.35,
                "depths": "2,2,2,2",
                "img_size": 64,
                "gray": 1,
                "dtype": "int8",
                "n": data.get("n"),
                "pred_match": data.get("pred_match"),
                "mean_cosine": data.get("mean_cosine"),
                "mean_latency_us": data.get("mean_latency_us"),
                "mean_latency_ms": round(float(data.get("mean_latency_us", 0)) / 1000.0, 3),
                "source": str(path.as_posix()),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} n={len(rows)}")
    if args.enum.exists():
        enum = json.loads(args.enum.read_text(encoding="utf-8"))
        print(f"enum ops ready: {enum.get('n')} (op-level mining pending)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
