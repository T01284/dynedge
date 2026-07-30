#!/usr/bin/env python3
"""Build NPU vs NCNN latency comparison for T527 v4 anchors."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "t527_v4"
NPU = OUT / "npu_matrix_summary.csv"
NCNN = OUT / "matrix_summary.csv"
CMP = OUT / "npu_vs_ncnn_latency.csv"


def main() -> None:
    npu = list(csv.DictReader(NPU.open(encoding="utf-8")))
    ncnn = list(csv.DictReader(NCNN.open(encoding="utf-8")))
    nc = {(r["model"], r["dtype"]): float(r["latency_us_p50"]) for r in ncnn}
    rows = []
    for r in npu:
        m = r["model"]
        npu_us = float(r["latency_us_p50"])
        fp = nc.get((m, "fp32"))
        i8 = nc.get((m, "int8"))
        rows.append(
            {
                "model": m,
                "tag": r.get("tag", ""),
                "input": r["input"],
                "npu_us_p50": int(npu_us),
                "npu_ms_p50": round(npu_us / 1000, 3),
                "ncnn_fp32_ms": round(fp / 1000, 3) if fp else "",
                "ncnn_int8_ms": round(i8 / 1000, 3) if i8 else "",
                "speedup_vs_int8": round(i8 / npu_us, 2) if i8 else "",
                "speedup_vs_fp32": round(fp / npu_us, 2) if fp else "",
            }
        )
        print(
            f"{rows[-1]['tag']:20s} NPU {rows[-1]['npu_ms_p50']:6.3f} ms | "
            f"INT8 {rows[-1]['ncnn_int8_ms']} ms | "
            f"x{rows[-1]['speedup_vs_int8']}"
        )
    with CMP.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"OK -> {CMP}")


if __name__ == "__main__":
    main()
