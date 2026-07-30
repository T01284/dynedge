#!/usr/bin/env python3
"""Push G5 host-decided exits to H618 and wall-clock vs T3/force-final.

Mirror of deploy/t527/scripts/run_g5_board_early_exit.py.
"""
from __future__ import annotations

import json
import re
import tarfile
import tempfile
from pathlib import Path

from ssh_h618 import put, run

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/h618_v4/g5_early_exit"
T3_NCNN = ROOT / "results/t527_v4/t3_early_exit/ncnn"
BIN = ROOT / "deploy/ncnn_edge/build-linux-aarch64-acc/fer_early_exit"
THREADS = 4


def parse_result(text: str) -> dict:
    m = re.search(r"^RESULT .+$", text, re.M)
    if not m:
        return {"raw": text}
    row = {"raw": m.group(0)}
    for p in m.group(0).split()[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        row[k] = v
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    exits_g5 = list(OUT.glob("exits_g5_h618_b5ms_n80.txt"))
    if not exits_g5:
        raise SystemExit("run: python deploy/t527/scripts/g5_controller_decide.py --device h618 --budget-ms 5")
    exits_path = exits_g5[0]
    exits_t3 = OUT / "exits_t3_tau0.85_n80.txt"
    if not exits_t3.exists():
        raise SystemExit(f"missing {exits_t3}")

    bins = sorted((ROOT / "results/t527_v4/eval500_bins/hw112").glob("*.bin"))[:80]
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "bins.tar"
        with tarfile.open(tar_path, "w") as tar:
            for p in bins:
                tar.add(p, arcname=p.name)
        run("rm -rf /root/h618_g5_ee; mkdir -p /root/h618_g5_ee/bins /root/h618_g5_ee/models")
        put(str(BIN), "/root/h618_g5_ee/fer_early_exit")
        put(str(T3_NCNN / "m.ncnn.param"), "/root/h618_g5_ee/models/m.ncnn.param")
        put(str(T3_NCNN / "m.ncnn.bin"), "/root/h618_g5_ee/models/m.ncnn.bin")
        put(str(tar_path), "/root/h618_g5_ee/bins.tar")
        put(str(exits_path), "/root/h618_g5_ee/exits_g5.txt")
        put(str(exits_t3), "/root/h618_g5_ee/exits_t3.txt")

    code, out, err = run(
        "cd /root/h618_g5_ee && tar xf bins.tar -C bins && "
        "for f in bins/*; do echo /root/h618_g5_ee/$f; done > list80.txt && "
        "chmod +x fer_early_exit && wc -l list80.txt exits_g5.txt exits_t3.txt",
        timeout=120,
    )
    print(out, err, "prep", code)
    if code != 0:
        return code

    common = (
        "./fer_early_exit --param models/m.ncnn.param --bin models/m.ncnn.bin "
        f"--list list80.txt --h 112 --w 112 --c 3 --threads {THREADS} "
        "--out0 out0 --out1 out1 --out2 out2 --out3 out3"
    )

    results = {}
    for name, extra in (
        ("force_final", " --force-final"),
        ("t3_online_threshold", " --tau 0.85"),
        ("g5_forced_exits", " --exits exits_g5.txt --tau 0.85"),
        ("t3_forced_exits", " --exits exits_t3.txt --tau 0.85"),
    ):
        code, o, e = run(f"cd /root/h618_g5_ee && {common}{extra}", timeout=300)
        print(name, o)
        if e:
            print(e[-400:])
        (OUT / f"board_{name}.txt").write_text((o or "") + "\n" + (e or ""), encoding="utf-8")
        results[name] = parse_result(o or "")
        if code != 0:
            return code

    def p50(row: dict) -> float:
        return float(row.get("p50_ms", "nan"))

    ff = p50(results["force_final"])
    summary = {
        "status": "ok_g5_board_pipe",
        "device": "h618",
        "note": (
            "G5 host C3 hard-gate + board --exits; budget grid imperfect; not paper-final"
        ),
        "threads": THREADS,
        "n": 80,
        "exits_g5_file": exits_path.name,
        "board": results,
        "speedup_p50": {
            "t3_online_vs_ff": round(ff / p50(results["t3_online_threshold"]), 4)
            if p50(results["t3_online_threshold"]) > 0
            else None,
            "g5_vs_ff": round(ff / p50(results["g5_forced_exits"]), 4)
            if p50(results["g5_forced_exits"]) > 0
            else None,
        },
    }
    (OUT / "g5_board_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
