#!/usr/bin/env python3
"""Enumerate operator configs for MCU/H618 LUT mining (v3_wide/v4 topology)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "lut") not in sys.path:
    sys.path.insert(0, str(ROOT / "lut"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topo_ops import dedupe_ops, expand_subnet_ops  # noqa: E402

WIDTHS = (0.35, 0.5, 0.75, 1.0)
RES = (64, 96, 112)
# Cover e2e matrix depths + max path for completeness
DEPTH_SETS = (
    (2, 2, 2, 2),
    (3, 3, 3, 3),
    (4, 4, 4, 4),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "lut" / "enum_v4_full.json")
    args = ap.parse_args()

    rows: list[dict] = []
    # MCU: w0.35 @64 int8 both chips (depths that matter for anchor + deeper LUT)
    for depths in DEPTH_SETS:
        for backend in ("esp32p4", "esp32s3"):
            rows += expand_subnet_ops(
                width=0.35,
                depths=depths,
                img_size=64,
                dtype="int8",
                backend=backend,
            )

    # H618: full width × res × fp32+int8 × depth sets
    for width in WIDTHS:
        for img in RES:
            for dtype in ("fp32", "int8"):
                for depths in DEPTH_SETS:
                    rows += expand_subnet_ops(
                        width=width,
                        depths=depths,
                        img_size=img,
                        dtype=dtype,
                        backend="h618_ncnn",
                    )

    uniq = dedupe_ops(rows)
    # counts
    by_be: dict[str, int] = {}
    for r in uniq:
        by_be[r["backend"]] = by_be.get(r["backend"], 0) + 1
    mineable = sum(1 for r in uniq if r["op"] in {"conv", "dwconv", "gemm"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "note": "v4 full enum: v3_wide topo; MCU w0.35@64; H618 all width×res×dtype; depths 2/3/4",
        "n": len(uniq),
        "mineable_conv_dw_gemm": mineable,
        "by_backend": by_be,
        "ops": uniq,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # also refresh firstpass alias pointer note
    print(f"wrote {args.out} n={len(uniq)} mineable={mineable} by_backend={by_be}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
