#!/usr/bin/env python3
"""Poll / pull T527 full LUT mine results."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "deploy" / "t527" / "scripts"))
from mine_ops_t527_full import META_FILE, OUT_CSV, REMOTE_WORK, merge_results  # noqa: E402
from ssh_t527 import run  # noqa: E402


def main() -> int:
    code, out, err = run(
        f"tail -n 8 {REMOTE_WORK}/mine.log 2>/dev/null; "
        f"wc -l {REMOTE_WORK}/results.csv 2>/dev/null; "
        f"ps aux | grep '[m]ine_all' || echo NOT_RUNNING",
        timeout=30,
    )
    print(out + err)
    done = "DONE ok=" in out
    running = "mine_all" in out and "NOT_RUNNING" not in out.splitlines()[-1]
    if not done and running:
        print("still running...")
        return 0
    if not done and not running:
        print("process ended without DONE — pulling partial")
    code2, csv_txt, err2 = run(f"cat {REMOTE_WORK}/results.csv", timeout=120)
    if code2 != 0:
        print(err2)
        return code2
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    merge_results(csv_txt, meta, OUT_CSV)
    rows = list(csv.DictReader(OUT_CSV.open(encoding="utf-8")))
    ok = sum(1 for r in rows if r.get("ok") == "1")
    print(f"summary ok={ok}/{len(rows)} -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
