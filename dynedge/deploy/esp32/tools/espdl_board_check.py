#!/usr/bin/env python3
"""Compare board RESULT logits vs host_ref.json (same 8 embedded images)."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import serial

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="SERIAL_PORT")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--ref",
        type=Path,
        default=ROOT / "deploy/mcu_espdl/models/d2222_ferplus_64_esp32p4_host_ref.json",
    )
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "deploy/mcu_espdl/models/d2222_ferplus_64_esp32p4_board_check.json",
    )
    args = ap.parse_args()

    ref = json.loads(args.ref.read_text(encoding="utf-8"))
    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(1.5)
    ser.reset_input_buffer()
    ser.write(b"CMD INFO\n")
    ser.flush()
    t0 = time.time()
    while time.time() - t0 < 5:
        line = ser.readline().decode("utf-8", "replace").rstrip()
        if line:
            print(line)
            if "READY" in line:
                break

    ser.write(b"CMD CHECK\n")
    ser.flush()
    results = []
    t0 = time.time()
    while time.time() - t0 < 60:
        line = ser.readline().decode("utf-8", "replace").rstrip()
        if not line:
            continue
        if "RESULT" in line and not line.startswith("RESULT "):
            line = line[line.find("RESULT ") :]
        print(line)
        if line.startswith("RESULT "):
            # RESULT kind=check idx=0 latency_us=.. nlogits=8 out_exp=-4 host_pred=0 logits=a,b,...
            m = re.search(
                r"idx=(\d+).*latency_us=(\d+).*out_exp=(-?\d+).*host_pred=(-?\d+).*logits=([-\d,]+)",
                line,
            )
            if not m:
                continue
            idx = int(m.group(1))
            lat = int(m.group(2))
            exp = int(m.group(3))
            host_pred = int(m.group(4))
            raw = [int(x) for x in m.group(5).split(",") if x]
            scale = 2.0**exp
            deq = (np.asarray(raw, dtype=np.float64) * scale).tolist()
            results.append(
                {
                    "index": idx,
                    "latency_us": lat,
                    "out_exp": exp,
                    "raw_logits": raw,
                    "deq_logits": deq,
                    "board_pred": int(np.argmax(raw)),
                    "host_pred_embedded": host_pred,
                }
            )
        if line.startswith("CHECK_DONE"):
            break
    ser.close()

    rows = []
    pred_match = 0
    cosines = []
    for r in results[: args.n]:
        h = ref["samples"][r["index"]]
        hq = np.asarray(h["int8_logits"], dtype=np.float64)
        bq = np.asarray(r["deq_logits"], dtype=np.float64)
        cos = float(np.dot(hq, bq) / (np.linalg.norm(hq) * np.linalg.norm(bq) + 1e-12))
        ok = int(h["int8_pred"] == r["board_pred"])
        pred_match += ok
        cosines.append(cos)
        rows.append(
            {
                **r,
                "host_int8_logits": h["int8_logits"],
                "host_int8_pred": h["int8_pred"],
                "pred_match": bool(ok),
                "cosine": cos,
            }
        )

    out = {
        "port": args.port,
        "n": len(rows),
        "pred_match": pred_match,
        "pred_match_rate": pred_match / max(1, len(rows)),
        "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "mean_latency_us": float(np.mean([r["latency_us"] for r in rows])) if rows else 0.0,
        "samples": rows,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"board_check pred_match={pred_match}/{len(rows)} "
        f"mean_cos={out['mean_cosine']:.4f} mean_lat_ms={out['mean_latency_us']/1000:.2f} "
        f"-> {args.out}"
    )
    return 0 if pred_match == len(rows) and out["mean_cosine"] > 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
