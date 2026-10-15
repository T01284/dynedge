#!/usr/bin/env python3
"""Collect B15 board results next to the B14 simulation -> results/stageB/mcu_board_acc/summary.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "results/stageB/mcu_board_acc"
SIM = ROOT / "results/stageB/mcu_offline_acc"


def main() -> int:
    sim = {}
    for ds in ("ferplus", "expw"):
        for r in csv.DictReader((SIM / f"summary_{ds}.csv").open(encoding="utf-8")):
            sim[(r["model"], r["target"], r["calib"], r["prep"])] = r
    rows = []
    for target in ("esp32p4", "esp32s3"):
        for js in sorted((BOARD / target).glob("*_pil.json")):
            model = js.name[: -len("_pil.json")]
            gate = json.loads((BOARD / target / f"{model}_gate8.json").read_text(encoding="utf-8"))
            row = {"target": target, "model": model, "gate8": gate["pred_match"],
                   "stream_max_dlogit": gate.get("stream_max_abs_dlogit")}
            for prep in ("pil", "train"):
                p = BOARD / target / f"{model}_{prep}.json"
                if not p.exists():
                    continue
                b = json.loads(p.read_text(encoding="utf-8"))
                s = sim.get((model, target, "deployed_pil", prep))
                row[f"board_{prep}"] = round(b["acc"], 4)
                row[f"sim_{prep}"] = round(float(s["acc"]), 4) if s else None
                row["n"] = b["n"]
                row["p50_ms"] = round(b["median_latency_ms"], 1)
                row["pytorch"] = round(float(s["pytorch_acc"]), 4) if s else None
            rows.append(row)
    fields = ["target", "model", "n", "pytorch", "board_pil", "sim_pil", "board_train", "sim_train",
              "p50_ms", "gate8", "stream_max_dlogit"]
    out = BOARD / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" ".join(f"{k}={r.get(k)}" for k in fields))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
