#!/usr/bin/env python3
"""H618 G5 remaining budgets board wall-clock (b10/b20/b50) vs force_final / T3.

Does NOT touch T527. Reuses /root/h618_g5_ee if present, else pushes fresh.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "deploy/t527/scripts"))

from ssh_h618 import put, run  # noqa: E402

OUT = ROOT / "results/h618_v4/g5_early_exit"
THREADS = 4
BUDGETS = (10, 20, 50)


def parse_result(text: str) -> dict:
    m = re.search(r"^RESULT .+$", text, re.M)
    if not m:
        return {"raw": text}
    row = {"raw": m.group(0)}
    for p in m.group(0).split()[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            row[k] = v
    return row


def host_decide(budget: int) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "deploy/t527/scripts/g5_controller_decide.py"),
        "--device",
        "h618",
        "--budget-ms",
        str(budget),
        "--n",
        "80",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))
    p = OUT / f"exits_g5_h618_b{budget}ms_n80.txt"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # ensure remote workspace from prior G5 run (or recreate lightly)
    code, out, err = run("test -x /root/h618_g5_ee/fer_early_exit && test -f /root/h618_g5_ee/list80.txt && echo OK || echo NEED", timeout=30)
    if "NEED" in (out or ""):
        print("remote workspace missing; run run_g5_board_early_exit.py once first")
        return 1
    print("remote", out.strip())

    exits = {}
    for b in BUDGETS:
        exits[b] = host_decide(b)
        put(str(exits[b]), f"/root/h618_g5_ee/exits_g5_b{b}.txt")

    common = (
        "./fer_early_exit --param models/m.ncnn.param --bin models/m.ncnn.bin "
        f"--list list80.txt --h 112 --w 112 --c 3 --threads {THREADS} "
        "--out0 out0 --out1 out1 --out2 out2 --out3 out3"
    )
    results: dict = {}
    # shared baselines once
    for name, extra in (
        ("force_final", " --force-final"),
        ("t3_tau0.85", " --tau 0.85"),
    ):
        code, o, e = run(f"cd /root/h618_g5_ee && {common}{extra}", timeout=300)
        print(name, o.strip())
        (OUT / f"board_budgetsuite_{name}.txt").write_text((o or "") + "\n" + (e or ""), encoding="utf-8")
        results[name] = parse_result(o or "")
        if code != 0:
            return code

    for b in BUDGETS:
        code, o, e = run(
            f"cd /root/h618_g5_ee && {common} --exits exits_g5_b{b}.txt --tau 0.85",
            timeout=300,
        )
        key = f"g5_b{b}ms"
        print(key, o.strip())
        (OUT / f"board_{key}.txt").write_text((o or "") + "\n" + (e or ""), encoding="utf-8")
        results[key] = parse_result(o or "")
        if code != 0:
            return code

    def p50(row: dict) -> float:
        return float(row.get("p50_ms", "nan"))

    ff = p50(results["force_final"])
    summary = {
        "status": "ok_h618_g5_budget_suite",
        "device": "h618",
        "note": "T527 frozen for P5S1; H618 G5 b10/20/50 lying-flat confirmation",
        "threads": THREADS,
        "n": 80,
        "board": results,
        "speedup_p50_vs_ff": {
            k: round(ff / p50(v), 4) if p50(v) > 0 else None for k, v in results.items()
        },
    }
    (OUT / "g5_budget_suite_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
