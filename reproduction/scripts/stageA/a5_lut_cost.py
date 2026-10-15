#!/usr/bin/env python3
"""A5: cost of building the latency LUTs, from the recorded mining protocol and logs.

Operator mining (Linux tiers): warmup 10 + 80 timed iters per operator shape
(lut/mine_ops_h618_batch.py; SUMMARY.md 8.7 for T527).
MCU: per-config flash + bench; inter-attempt spacing from lut/mcu_e2e_failures.csv timestamps.

Outputs: results/stageA/a5_cost_table.csv, a5_summary.json
"""
from __future__ import annotations

import json

import pandas as pd

from common import OUT, ROOT

WARMUP, ITERS = 10, 80


def main() -> int:
    h = pd.read_csv(ROOT / "lut/ops_h618_v4_full.csv")
    t = pd.read_csv(ROOT / "lut/ops_t527_v4_full.csv")
    h_ok = h[h.ok.astype(str).str.lower().isin(["true", "1"])]
    t_ok = t[t.ok.astype(str).isin(["1", "True", "true"])]

    def op_minutes(df, col):
        return float(df[col].sum() * (WARMUP + ITERS) / ITERS / 6e7)

    rows = [
        {"tier": "H618 NCNN operator LUT", "unit": "operator shape",
         "enumerated": 706, "measured_ok": int(len(h_ok)), "failed": int(706 - len(h_ok)),
         "protocol": "warmup 10 + iters 80, 4 threads",
         "bench_minutes": round(op_minutes(h_ok, "p50_us"), 2),
         "extra": "pure timed-iteration sum; wall clock is dominated by per-shape ncnn2table/ncnn2int8 "
                  "(INT8, KL calib) and process spawn, logged only as 'a few minutes' (SUMMARY 8.7). "
                  "88 add/gap unsupported (proxy in prediction)"},
        {"tier": "T527 NCNN operator LUT", "unit": "operator shape",
         "enumerated": 706, "measured_ok": int(len(t_ok)), "failed": 0,
         "protocol": "warmup 10 + iters 80, 4 threads",
         "bench_minutes": round(op_minutes(t_ok, "latency_us_p50"), 2),
         "extra": "pure timed-iteration sum; host pack+INT8 quant ~3.5 min and per-shape quantisation "
                  "dominate the wall clock ('board mine: a few minutes', SUMMARY 8.7). "
                  "88 int8 gemm/add/gap are fp32 proxies"},
    ]

    mcu = pd.read_csv(ROOT / "lut/e2e_mcu_truth.csv")
    fails = pd.read_csv(ROOT / "lut/mcu_e2e_failures.csv", parse_dates=["ts"])
    gaps = fails.ts.sort_values().diff().dt.total_seconds().dropna()
    gaps = gaps[(gaps > 30) & (gaps < 900)]
    per_attempt_min = float(gaps.median() / 60)
    for dev, name in (("esp32p4", "ESP32-P4"), ("esp32s3", "ESP32-S3")):
        n_ok = int((mcu.device == dev).sum())
        n_fail = int(((fails.device == dev) & (~fails.reason.str.startswith("skipped"))).sum())
        rows.append({"tier": f"{name} exhaustive e2e LUT", "unit": "subnet (flash + bench)",
                     "enumerated": n_ok + n_fail, "measured_ok": n_ok, "failed": n_fail,
                     "protocol": "per subnet: export, ESP-DL PTQ, flash, on-device bench",
                     "bench_minutes": round((n_ok + n_fail) * per_attempt_min, 1),
                     "extra": f"~{per_attempt_min:.1f} min/attempt (median gap between the 25 logged attempts, "
                              f"07-21 14:53-15:51); includes failed flashes"})

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "a5_cost_table.csv", index=False)
    summary = {
        "operator_protocol": {"warmup": WARMUP, "iters": ITERS, "threads": 4,
                              "source": "lut/mine_ops_h618_batch.py L263; SUMMARY.md 8.7"},
        "h618_operator_median_us": float(h_ok.p50_us.median()),
        "t527_operator_median_us": float(t_ok.latency_us_p50.median()),
        "mcu_attempt_gap_s_median": float(gaps.median()),
        "mcu_attempt_gap_s_n": int(len(gaps)),
        "total_bench_hours": round(float(tab.bench_minutes.sum()) / 60, 2),
        "note": "Bench hours count on-device measurement time only. Host-side export, PTQ and transfer "
                "are extra and not in these logs except the T527 pack+quant note.",
    }
    (OUT / "a5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(tab.to_string(index=False))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
